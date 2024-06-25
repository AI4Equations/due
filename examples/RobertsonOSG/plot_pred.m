load("full_true.csv")
load("model/pred.mat")
load("full_t_100000.csv")
semilogx(full_t_100000, squeeze(trajectories(1,1,:)), 'LineStyle', 'none', 'Marker','s')
hold
semilogx(full_true(1:904,1), full_true(1:904,2))%837%893
semilogx(full_t_100000, 10^4*squeeze(trajectories(1,2,:)), 'LineStyle', 'none', 'Marker','*')
semilogx(full_true(1:904,1), full_true(1:904,3))
semilogx(full_t_100000, squeeze(trajectories(1,3,:)), 'LineStyle', 'none', 'Marker','o')
semilogx(full_true(1:904,1), full_true(1:904,4))
