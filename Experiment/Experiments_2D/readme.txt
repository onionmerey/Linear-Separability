At the beginning of our work we created 2D sets of points and calculated their distances. In this experiment we tested sampling method and Linear Programming on those datasets wiht known distances. 
Objective: 
1) verify the theoretical guarantee from the paper "Property Testing of LP-type problems" that sample size of 10*d/epsilon and LP with extra cheking provides 2/3 accuracy, 
2) see how tester behaves for epsilon far cases
Code logic: 
1) The inputs are read from .txt file with pattern name "2D_numberOfPoints_trueDistance_randomId". File format (first line number of points then each row is "x y z " where x, y - coordinates, z label (. or #))
2) The only hyperparameter is k_factor. This k_factor determines the sample size k_factor*d/epsilon (d = 2 for that experiment)
3) The code runs LP on all points and fins whether points are linearly separable or not indeed (this used as ground truth separability furhter to compare with answer of LP on sampled points)
4) Two sampling approaches are applied. Method 1 is "Sample points. Try to find whether they are linearly  separable by LP. So actually they are separable if LP can find some feasible solution"
5) Method 2 is "Do what is done in Method 1. If LP found some feasible solution, then, sample another 2/epsilon points (different from initially sampled points if possible) and check whether those points violate the solution found in previous step."
Both methods do 100 iteration for each file and epsilon combination. 
6) All results with measured time saved in .csv files
7) .csv files are named "LP_M1_vs_M2_k1_20250903_184059.csv". It means that this file is the result of experiment done on NLS (non-linearly separable) files with k_factor = 1. Here number of points starts from 100 and gradually grows with different steps until 100000. There some files with 500r and 600k points.
8) The files named "LP_M1_vs_M2_generated_multiN_k2_20250904_222139.csv" are results of experiment done on LS cases with k_factor = 2. Here number of points starts from 5000 and grows until 100000 wiht step 5000. This experiment is done by generating points and testing without saving. For generation the generation method from "a new sufficient and necessary condition for testing linear separability between two sets" paper was used. 
Results:
1) Results show that Epstein's approach gives a very good boost in accuracy for NLS cases even for very small sample cases (4 points -> ~50 accuacy). However,it also has very low accuracy for LS cases when sample size is very small.
2) I think in Epstein's approach the contructed solution from samples is in most cases is far from optimal vector of separation and it does him a favor in NLS case. I mean 4 points construct very bad solution then this bad solution will violate any points samled in 2 step and say that answer is NLS.
However, for LS case it became its curse. 4 points provide very bad solution that violates the new points even if overall two sets are separable. 
3) The Method 1 showed its 1-sided error feature. In LS cases always says LS. 
4) For LS cases, Epstein's approach showed >2/3 accuracy with 10*d/epsilon samples for all epsilons
5) The last is mostly true for 5*d/epsilon also.
