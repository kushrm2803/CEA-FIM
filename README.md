# CEA-FIM: Community-based Evolutionary Algorithm for Fair Influence Maximization

This repository contains an enhanced implementation of the Community-based Evolutionary Algorithm for Fair Influence Maximization (CEA-FIM). We introduce significant improvements to the base algorithm from the [original paper](https://ieeexplore.ieee.org/abstract/document/10542566) to better address fairness concerns in social network influence maximization.

## Algorithm Variants

### Enhanced PageRank Branch (kush)

#### Changes Introduced
1. Replaced standard PageRank with Fairness-Biased PageRank
2. Added custom personalization vector to favor under-represented groups

#### How It Works
- Computes group sizes: `group_sizes = {val: len(nodes) for val, nodes in nodes_attr.items()}`
- Node personalization weight:
```math
p_i = \frac{1}{|R_j|} \text{ if node } i \text{ belongs to group } R_j
```
- This gives smaller demographic groups higher initial weights

#### Benefits
- Counteracts standard PageRank's bias toward large, dense communities
- Balances structural influence with demographic fairness
- Improves visibility of nodes from minority communities

### Enhanced Scoring Branch (main)

#### Changes Introduced
1. Bridge-aware community scoring with formula:
```math
S'_C = S_C \times (1 + \beta_t)
```
2. Multi-dimensional fairness personalization incorporating:
   - Group size
   - Influence potential 
   - Selection deficit

#### Technical Details
- Uses Participation Coefficient (PC) to detect bridge nodes:
```math
PC(i) = 1 - \sum_s (\frac{k_{i,s}}{k_i})^2
```
- Enhanced node weight calculation:
```math
Weight_{enhanced}(i) = Weight_{base}(i) \times (1 + \alpha_{bridge} \times PC(i) \times (1 + D_i))
```

#### Benefits
- Better representation of multiple protected groups
- Maintains strong influence potential
- More balanced diffusion across different communities
- Prevents influence concentration in single attribute groups

### Original Algorithm (original branch)
Contains the baseline implementation from the original paper.

## Reference
Our enhancements are based on the algorithm presented in:
[Community-based Evolutionary Algorithm for Fair Influence Maximization](https://ieeexplore.ieee.org/abstract/document/10542566)

## Team Members

| Name                | Roll Number         | GitHub ID           |
| :------------------ | :------------------ | :------------------ |
| Kush Mahajan         | 2022CSB1089       | [@kushrm2803](https://github.com/kushrm2803)      |
| Swapnil Pandey   | 2022CSB1133 | [@SwapnilPandey2210](https://github.com/SwapnilPandey2210)      |
| Ayush Patel   | 2022CSB1101 | [@Gujju-atWork](https://github.com/Gujju-atWork)       |
| Nishant Patil   | 2022CSB1097 | [@Nishant984](https://github.com/Nishant984)       |
