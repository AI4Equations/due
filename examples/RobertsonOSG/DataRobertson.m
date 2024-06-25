rng(0)
% y0 = [1; 0; 0];
% tspan = [0 5*logspace(-5,6,1067)];
M = [1 0 0; 0 1 0; 0 0 0];
options = odeset('Mass',M,'RelTol',1e-4,'AbsTol',[1e-6 1e-10 1e-6]);
% [t,y] = ode15s(@Robertsondae,tspan,y0,options);
% y(:,2) = 1e4*y(:,2);
% semilogx(t,y);
% ylabel('1e4 * y(:,2)');
% title('Robertson DAE problem with a Conservation Law, solved by ODE15S');
% asd

N = 100000;
trajectories = zeros(N,3,2);
dt = zeros(N,1);
for n = 1:N
    n
    % random initial state, with sum=1 constraint
    a = rand;
    c = rand;
    aa = a/(a+c);
    cc = c/(a+c);
    a  = aa;
    c  = cc;
    b = rand;
    b = b*5e-5;
    y0 = [a/(a+b+c); b/(a+b+c); c/(a+b+c)];
    trajectories(n,:,1) = y0;

    % random time stepsize, sampling in the logspace of the interval [1e-6,1e6]
    z = rand;
%     z = (3-sqrt(9-8*z))/2;
    delta = 10^(8*z-4.5);
    dt(n,1) = delta;
    [t,y] = ode15s(@Robertsondae, [0,delta], y0, options);
%     y(:,2) = 1e4*y(:,2);
%     semilogx(t,y);
%     ylabel('1e4 * y(:,2)');
%     title('Robertson DAE problem with a Conservation Law, solved by ODE15S');
    l = size(y);
    l = l(1);
    y = y(l,:);
    trajectories(n,:,2) = y;
end
save("RobertsonOSG_train.mat", "trajectories", "dt")
histogram(log10(dt));
%set(gca,'xscale','log');
% y(:,2) = 1e4*y(:,2);
% semilogx(t,y);
% ylabel('1e4 * y(:,2)');
% title('Robertson DAE problem with a Conservation Law, solved by ODE15S');
