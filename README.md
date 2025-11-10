# CEA-FIM: Enhanced PageRank Branch

This repository branch (`kush`) contains an enhanced implementation of the Community-based Evolutionary Algorithm for Fair Influence Maximization (CEA-FIM).

This variant focuses on improving fairness by replacing the standard PageRank component with a **Fairness-Biased PageRank** and a custom personalization vector.

## Algorithm Details

#### Changes Introduced
1.  Replaced standard PageRank with Fairness-Biased PageRank
2.  Added custom personalization vector to favor under-represented groups

#### How It Works

The core enhancement is the introduction of a fairness-aware personalization vector for PageRank.

-   First, compute group sizes based on the protected attribute:
    `group_sizes = {val: len(nodes) for val, nodes in nodes_attr.items()}`

-   Then, calculate the node personalization weight. Nodes in smaller demographic groups receive a higher initial weight:
    ```math
    p_i = \frac{1}{|R_j|} \text{ if node } i \text{ belongs to group } R_j
    ```

#### Benefits

-   **Counteracts Bias:** Actively counteracts standard PageRank's inherent bias toward large, dense communities.
-   **Balances Influence:** Strikes a better balance between a node's structural influence and demographic fairness.
-   **Improves Visibility:** Enhances the visibility and influence potential of nodes from under-represented or minority communities.

## Reference

Our enhancements are based on the algorithm presented in:
[Community-based Evolutionary Algorithm for Fair Influence Maximization](https://ieeexplore.ieee.org/abstract/document/10542566)
