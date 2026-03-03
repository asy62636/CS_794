"""
Prefix Cache with Radix Tree for KV Cache Sharing
"""
import torch

class RadixNode:
    """
    A node in the radix tree.
    Each node represents a sequence of tokens on an edge.
    """
    
    def __init__(self, tokens=None):
        self.tokens = tokens if tokens is not None else []
        self.kv_cache = None  # Incremental KV cache for ONLY these tokens
        self.children = {}  # Dict: first_token -> RadixNode

    def __repr__(self):
        cache_info = f"cache_len={self.kv_cache[0][0].shape[2]}" if self.kv_cache else "no_cache"
        return f"Node(tokens={self.tokens}, {cache_info}, children={len(self.children)})"


class PrefixCache:
    """
    Radix tree for storing and computing KV caches with prefix sharing.
    
    Three-phase operation:
    1. Build tree structure (insert_structure_only)
    2. Compute all KV caches (compute_all_kv_caches)
    3. Retrieve KV for generation (get_kv_for_prompt)
    """
    
    def __init__(self):
        self.root = RadixNode()
        
        # Statistics
        self.total_queries = 0
        self.total_tokens_saved = 0
        self.total_tokens_computed = 0

    # ========================================================================
    # PHASE 1: BUILD TREE STRUCTURE (NO KV COMPUTATION)
    # ========================================================================
    
    def insert_structure_only(self, tokens):
        """
        Insert token sequence into tree WITHOUT computing KV cache.
        Just builds the tree structure for prefix sharing.
        
        Args:
            tokens: List of token IDs
        """
        if not tokens:
            print("⚠️  Empty token sequence, skipping")
            return
        
        print(f"\n[INSERT] Adding tokens to tree: {tokens}")
        
        current_node = self.root
        pos = 0

        while pos < len(tokens):
            first_token = tokens[pos]
            
            # Case 1: No child exists with this first token
            if first_token not in current_node.children:
                remaining = tokens[pos:]
                new_node = RadixNode(remaining)
                current_node.children[first_token] = new_node
                print(f"  ✓ Created new leaf: {remaining}")
                return
            
            child = current_node.children[first_token]
            print(f"  Found existing child: {child.tokens}")
            
            # Find how many tokens match with this child
            match_len = 0
            for i in range(min(len(child.tokens), len(tokens) - pos)):
                if child.tokens[i] == tokens[pos + i]:
                    match_len += 1
                else:
                    break
            
            print(f"    Match length: {match_len}/{len(child.tokens)}")

            # Case 2: Full match - continue deeper in the tree
            if match_len == len(child.tokens):
                pos += match_len
                current_node = child
                print(f"    Full match, continuing deeper...")
            
            # Case 3: Partial match - need to split the node
            elif match_len > 0:
                print(f"    Partial match detected, splitting...")
                
                # 1. Extract the common prefix
                common_prefix = child.tokens[:match_len]
                
                # 2. Create new node for the common prefix
                common_node = RadixNode(common_prefix)
                
                # 3. Create node for the existing child's suffix
                child_suffix = child.tokens[match_len:]
                old_suffix_node = RadixNode(child_suffix)
                old_suffix_node.children = child.children  # Keep existing children
                
                # 4. Add old suffix as a child of the common node
                common_node.children[child_suffix[0]] = old_suffix_node
                
                # 5. Handle our remaining tokens
                our_remaining = tokens[pos + match_len:]
                
                if our_remaining:
                    # We have more tokens - create a new branch
                    new_suffix_node = RadixNode(our_remaining)
                    common_node.children[our_remaining[0]] = new_suffix_node
                    print(f"      Common: {common_prefix}")
                    print(f"      Old suffix: {child_suffix}")
                    print(f"      New suffix: {our_remaining}")
                else:
                    # Our tokens end exactly at the common prefix
                    print(f"      Common: {common_prefix} (ends here)")
                    print(f"      Old suffix: {child_suffix}")
                
                # 6. Replace the old child with the new common node
                current_node.children[first_token] = common_node
                return
            
            else:
                # match_len == 0, shouldn't happen since we checked first_token
                print(f"⚠️  Unexpected: match_len is 0")
                return
        
        print(f"  ✓ Consumed all tokens")

    # ========================================================================
    # PHASE 2: COMPUTE KV CACHES (DFS TRAVERSAL)
    # ========================================================================
    
    def compute_all_kv_caches(self, model):
        """
        Walk the tree and compute KV caches for all nodes.
        Each node's KV is computed exactly once at the correct position.
        
        Args:
            model: GPT model instance
        """
        print(f"\n{'='*70}")
        print("PHASE 2: COMPUTING KV CACHES FOR TREE")
        print(f"{'='*70}")
        
        device = next(model.parameters()).device
        
        # Start DFS from root
        self._compute_node_kv(model, self.root, past_kvs=None, position=0, depth=0, device=device)
        
        print(f"{'='*70}")
        print(f"KV CACHE COMPUTATION COMPLETE")
        print(f"Total tokens computed: {self.total_tokens_computed}")
        print(f"{'='*70}\n")

    def _compute_node_kv(self, model, node, past_kvs, position, depth, device):
        """
        Recursively compute KV cache for a node and its children.
        
        Args:
            model: GPT model
            node: Current RadixNode
            past_kvs: KV cache accumulated from ancestors
            position: Starting position index for this node's tokens
            depth: Tree depth (for indentation)
            device: Device for tensors
        """
        indent = "  " * depth
        
        # Root node has no tokens
        if not node.tokens:
            print(f"{indent}[ROOT]")
            # Recurse to all children
            for child in node.children.values():
                self._compute_node_kv(model, child, past_kvs, position, depth + 1, device)
            return
        
        num_tokens = len(node.tokens)
        print(f"{indent}[NODE] tokens={node.tokens}, position={position}")
        
        # Prepare input tensor
        tokens_tensor = torch.tensor([node.tokens], dtype=torch.long, device=device)
        
        # Forward pass to compute KV for these tokens
        print(f"{indent}  Computing KV...")
        with torch.no_grad():
            _, _, new_kvs = model(tokens_tensor, past_kvs=past_kvs, position_offset=position)
        
        # Store the incremental KV for this node
        node.kv_cache = new_kvs
        self.total_tokens_computed += num_tokens
        
        print(f"{indent}  ✓ Stored KV (shape: {new_kvs[0][0].shape})")
        
        # Concatenate with parent's KV for passing to children
        if past_kvs is not None:
            combined_kvs = self._concatenate_kv_caches([past_kvs, new_kvs])
        else:
            combined_kvs = new_kvs
        
        # Recurse to children with accumulated KV
        new_position = position + num_tokens
        for child in node.children.values():
            self._compute_node_kv(model, child, combined_kvs, new_position, depth + 1, device)

    # ========================================================================
    # PHASE 3: RETRIEVE KV FOR GENERATION
    # ========================================================================
    
    def get_kv_for_prompt(self, tokens):
        """
        Retrieve the accumulated KV cache for a given token sequence.
        
        Args:
            tokens: Token sequence (list of IDs)
            
        Returns:
            (accumulated_kv, position_offset)
            - accumulated_kv: Concatenated KV cache for the matched prefix
            - position_offset: Number of tokens matched (start position for remaining)
        """
        self.total_queries += 1
        
        print(f"\n[GET_KV] Retrieving for tokens: {tokens}")
        
        current_node = self.root
        accumulated_kvs = []
        matched_tokens = []
        pos = 0
        
        while pos < len(tokens):
            first_token = tokens[pos]
            
            if first_token not in current_node.children:
                print(f"  No child for token {first_token}, stopping")
                break
            
            child = current_node.children[first_token]
            print(f"  Checking child: {child.tokens}")
            
            # Check how many tokens match
            match_len = 0
            for i in range(min(len(child.tokens), len(tokens) - pos)):
                if child.tokens[i] == tokens[pos + i]:
                    match_len += 1
                else:
                    break
            
            print(f"    Matched: {match_len}/{len(child.tokens)} tokens")
            
            if match_len == 0:
                break
            
            # Collect this node's KV
            if child.kv_cache is not None:
                accumulated_kvs.append(child.kv_cache)
                print(f"    ✓ Collected KV (shape: {child.kv_cache[0][0].shape})")
            else:
                print(f"    ⚠️  No KV cache at this node")
            
            matched_tokens.extend(child.tokens[:match_len])
            pos += match_len
            
            # Continue deeper if full match
            if match_len == len(child.tokens):
                current_node = child
            else:
                # Partial match, stop here
                break
        
        # Concatenate all collected KVs
        if accumulated_kvs:
            combined = self._concatenate_kv_caches(accumulated_kvs)
            print(f"  ✓ Combined KV cache (shape: {combined[0][0].shape})")
        else:
            combined = None
            print(f"  No KV cache found")
        
        self.total_tokens_saved += len(matched_tokens)
        
        print(f"  Matched tokens: {matched_tokens}")
        print(f"  Position offset: {pos}")
        print(f"  Remaining: {tokens[pos:]}")
        
        return combined, pos

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _concatenate_kv_caches(self, kv_cache_list):
        """
        Concatenate multiple KV caches along the sequence dimension.
        
        Args:
            kv_cache_list: List of past_kvs (each is a list of layer (K, V) tuples)
            
        Returns:
            Combined past_kvs
        """
        if not kv_cache_list:
            return None
        
        if len(kv_cache_list) == 1:
            return kv_cache_list[0]
        
        num_layers = len(kv_cache_list[0])
        combined = []
        
        for layer_idx in range(num_layers):
            # Gather K and V tensors for this layer from all caches
            k_tensors = [cache[layer_idx][0] for cache in kv_cache_list]
            v_tensors = [cache[layer_idx][1] for cache in kv_cache_list]
            
            # Concatenate along sequence dimension (dim=2)
            combined_k = torch.cat(k_tensors, dim=2)
            combined_v = torch.cat(v_tensors, dim=2)
            
            combined.append((combined_k, combined_v))
        
        return combined
    
    def _slice_kv_cache(self, kv_cache, start_pos, end_pos):
        """
        Slice KV cache to keep only positions [start_pos:end_pos].
        
        Args:
            kv_cache: List of (K, V) tuples, one per layer
            start_pos: Starting position (inclusive)
            end_pos: Ending position (exclusive)
        
        Returns:
            Sliced KV cache
        """
        sliced = []
        for k, v in kv_cache:
            # K and V shape: [B, num_heads, seq_len, head_dim]
            # Slice along seq_len dimension (dim=2)
            sliced_k = k[:, :, start_pos:end_pos, :]
            sliced_v = v[:, :, start_pos:end_pos, :]
            sliced.append((sliced_k, sliced_v))
        
        return sliced

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def print_tree(self, node=None, indent=0):
        """
        Print the tree structure for debugging.
        
        Args:
            node: Starting node (default: root)
            indent: Indentation level
        """
        if node is None:
            print("\n" + "="*60)
            print("TREE STRUCTURE")
            print("="*60)
            node = self.root
        
        indent_str = "  " * indent
        
        if not node.tokens:
            print(f"{indent_str}[ROOT]")
        else:
            cache_info = f" (KV: {node.kv_cache[0][0].shape[2]} tokens)" if node.kv_cache else " (no KV)"
            print(f"{indent_str}└── {node.tokens}{cache_info}")
        
        for child in node.children.values():
            self.print_tree(child, indent + 1)
        
        if indent == 0:
            print("="*60 + "\n")
    
    def stats(self):
        """Print cache statistics."""
        if self.total_queries == 0:
            print("No queries yet")
            return
        
        avg_saved = self.total_tokens_saved / self.total_queries
        
        print(f"\n{'='*60}")
        print(f"PREFIX CACHE STATISTICS")
        print(f"{'='*60}")
        print(f"Total tokens computed: {self.total_tokens_computed}")
        print(f"Total queries:         {self.total_queries}")
        print(f"Total tokens saved:    {self.total_tokens_saved}")
        print(f"Avg tokens saved:      {avg_saved:.1f} per query")
        if self.total_tokens_computed > 0:
            efficiency = (self.total_tokens_saved / self.total_tokens_computed) * 100
            print(f"Efficiency:            {efficiency:.1f}% tokens reused")
        print(f"{'='*60}\n")
    
    def clear(self):
        """Clear all cached data."""
        self.root = RadixNode()
        self.total_queries = 0
        self.total_tokens_saved = 0
        self.total_tokens_computed = 0


def test_radix_tree_structure():
    """
    Test building radix tree structure without KV computation.
    """
    print("\n" + "="*70)
    print("TESTING RADIX TREE CONSTRUCTION")
    print("="*70)
    
    # Create cache
    cache = PrefixCache()
    
    # Batch of token sequences
    batch = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 6],
        [1, 7, 8, 9],
        [1, 7, 8, 9, 10],
        [10, 11, 12, 13],
        [10, 11, 12, 15, 6]
    ]
    
    print(f"\nBatch of {len(batch)} sequences:")
    for i, tokens in enumerate(batch):
        print(f"  Sequence {i+1}: {tokens}")
    
    print("\n" + "="*70)
    print("PHASE 1: BUILDING TREE STRUCTURE")
    print("="*70)
    
    # Insert all sequences
    for i, tokens in enumerate(batch):
        print(f"\n{'='*70}")
        print(f"INSERTING SEQUENCE {i+1}/{len(batch)}: {tokens}")
        print(f"{'='*70}")
        cache.insert_structure_only(tokens)
    
    # Print final tree
    print("\n" + "="*70)
    print("FINAL TREE STRUCTURE")
    print("="*70)
    cache.print_tree()
    
    # Print expected structure
    print("\n" + "="*70)
    print("EXPECTED TREE STRUCTURE:")
    print("="*70)
    print("""
root
 ├── [1]
 │    ├── [2]
 │    │    ├── [3]
 │    │    │    ├── [4]
 │    │    │    └── [5]
 │    │    └── [6]
 │    └── [7, 8, 9]
 │         └── [10]
 └── [10, 11, 12, 13]
""")
    
    # Verify some properties
    print("="*70)
    print("VERIFICATION:")
    print("="*70)
    
    # Check root has 2 children
    print(f"Root children count: {len(cache.root.children)} (expected: 2)")
    assert len(cache.root.children) == 2, "Root should have 2 children"
    
    # Check [1] exists
    assert 1 in cache.root.children, "Root should have child starting with token 1"
    node_1 = cache.root.children[1]
    print(f"✓ Node [1] exists with tokens: {node_1.tokens}")
    
    # Check [1] has 2 children: [2] and [7,8,9]
    print(f"Node [1] children count: {len(node_1.children)} (expected: 2)")
    assert len(node_1.children) == 2, "Node [1] should have 2 children"
    
    # Check [10,11,12,13] exists as separate branch
    assert 10 in cache.root.children, "Root should have child starting with token 10"
    node_10 = cache.root.children[10]
    print(f"✓ Node [10,11,12,13] exists with tokens: {node_10.tokens}")
    
    # Navigate to [1]->[2]->[3]
    assert 2 in node_1.children, "Node [1] should have child [2]"
    node_2 = node_1.children[2]
    print(f"✓ Node [2] exists with tokens: {node_2.tokens}")
    
    assert 3 in node_2.children, "Node [2] should have child [3]"
    node_3 = node_2.children[3]
    print(f"✓ Node [3] exists with tokens: {node_3.tokens}")
    
    # Check [3] has 2 children: [4] and [5]
    print(f"Node [3] children count: {len(node_3.children)} (expected: 2)")
    assert len(node_3.children) == 2, "Node [3] should have 2 children ([4] and [5])"
    
    assert 4 in node_3.children, "Node [3] should have child [4]"
    assert 5 in node_3.children, "Node [3] should have child [5]"
    print(f"✓ Node [3] has children [4] and [5]")
    
    # Navigate to [1]->[7,8,9]
    assert 7 in node_1.children, "Node [1] should have child [7,8,9]"
    node_789 = node_1.children[7]
    print(f"✓ Node [7,8,9] exists with tokens: {node_789.tokens}")
    
    # Check [7,8,9] has 1 child: [10]
    assert len(node_789.children) == 1, "Node [7,8,9] should have 1 child"
    assert 10 in node_789.children, "Node [7,8,9] should have child [10]"
    print(f"✓ Node [7,8,9] has child [10]")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    
    return cache


def visualize_sharing():
    """
    Visualize what gets shared between sequences.
    """
    print("\n" + "="*70)
    print("PREFIX SHARING ANALYSIS")
    print("="*70)
    
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 6],
        [1, 7, 8, 9],
        [1, 7, 8, 9, 10],
        [10, 11, 12, 13],
        [10, 11, 12, 15, 6]
    ]
    
    print("\nShared prefixes:")
    print("-" * 70)
    
    # Sequence 1 & 2
    print("Seq 1 [1,2,3,4] & Seq 2 [1,2,3,5]:")
    print("  Shared: [1,2,3] (3 tokens)")
    print("  Savings: 3 tokens reused for Seq 2")
    
    # Sequence 1 & 3
    print("\nSeq 1 [1,2,3,4] & Seq 3 [1,2,6]:")
    print("  Shared: [1,2] (2 tokens)")
    print("  Savings: 2 tokens reused for Seq 3")
    
    # Sequence 1 & 4
    print("\nSeq 1 [1,2,3,4] & Seq 4 [1,7,8,9]:")
    print("  Shared: [1] (1 token)")
    print("  Savings: 1 token reused for Seq 4")
    
    # Sequence 4 & 5
    print("\nSeq 4 [1,7,8,9] & Seq 5 [1,7,8,9,10]:")
    print("  Shared: [1,7,8,9] (4 tokens)")
    print("  Savings: 4 tokens reused for Seq 5")
    
    # Sequence 6
    print("\nSeq 6 [10,11,12,13]:")
    print("  Shared: None (independent branch)")
    print("  Savings: 0 tokens")
    
    # Total computation
    print("\n" + "="*70)
    print("COMPUTATION ANALYSIS:")
    print("="*70)
    
    total_tokens = sum(len(seq) for seq in sequences)
    print(f"Total tokens in all sequences: {total_tokens}")
    
    # Count unique nodes in tree
    unique_nodes = [
        "[1]",
        "[2]", "[7,8,9]",
        "[3]", "[6]", "[10]",
        "[4]", "[5]",
        "[10,11,12]","[13]", "[15,6]"
    ]
    tokens_computed = 1 + 1 + 3 + 1 + 1 + 1 + 1 + 1 + 4
    print(f"Tokens actually computed: {tokens_computed}")
    print(f"Tokens saved: {total_tokens - tokens_computed}")
    print(f"Efficiency: {((total_tokens - tokens_computed) / total_tokens * 100):.1f}% reduction")
    

if __name__ == "__main__":
    # Test tree construction
    cache = test_radix_tree_structure()
    
    # Show sharing analysis
    visualize_sharing()
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)