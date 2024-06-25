N = 359;
dt = 3000*ones(N,1);
dt(1,1) = 5e-5;
for i=2:26
    dt(i,1)=dt(i-1,1) * 2;
end

t = zeros(N+1,1);
for i = 1:N
    t(i+1,1)=t(i,1)+dt(i,1);
end
