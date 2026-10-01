# neural-PID
Neural Amortisation of an approach to Partial Information Decomposition\

Training results:\

======================================================================
TEST PERFORMANCE - TRUE PID ATOMS ACCURACY
======================================================================
Redundancy           MAE=4.8107444742e-03 | RMSE=7.4894723892e-03
Unique 1             MAE=5.0074758048e-03 | RMSE=8.4497798353e-03
Unique 2             MAE=5.3807032092e-03 | RMSE=9.6276350265e-03
Synergy              MAE=6.9487707359e-03 | RMSE=1.3152894183e-02

======================================================================
TEST PERFORMANCE - NULL STATISTICS (MU & SIGMA)
======================================================================
Redundancy           Mu MAE=2.9196019717e-03 | Sigma MAE=2.1728143322e-03
Unique 1             Mu MAE=6.3710142436e-03 | Sigma MAE=2.9267353170e-03
Unique 2             Mu MAE=7.5069443821e-03 | Sigma MAE=2.9668157049e-03
Synergy              Mu MAE=2.8335172988e-03 | Sigma MAE=2.3027225014e-03




Package dependencies: \
https://github.com/robince/gcmi \
scipy \
numpy

References \
Ince RA. Measuring multivariate redundant information with pointwise common change in surprisal. Entropy. 2017 Jun 29;19(7):318. \
Ince RA, Giordano BL, Kayser C, Rousselet GA, Gross J, Schyns PG. A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula. Human brain mapping. 2017 Mar;38(3):1541-73.

