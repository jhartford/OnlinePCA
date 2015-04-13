% for n = 100:200:1000
%     tic
%     rng(123)
%     m = 5000;
%     d = 50;
%     rho = 5;
%     n
%     Xt_f = strcat('data/Xt-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
%     X_f = strcat('data/X-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
%     y_f = strcat('data/y-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
%     [X, y] = gendata(m, n, d, rho);
%     Xt = PCA1(X, d, 0.35);
%     csvwrite(Xt_f, Xt(:,1:d));
%     csvwrite(y_f, y);
%     csvwrite(X_f, X);
%     clear all;
%     toc
% end


for d = 20:20:100
    rng(123)
    m = 5000;
    n = 400;
    rho = 5;
    d
    Xt_f = strcat('data/Xt-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    X_f = strcat('data/X-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    y_f = strcat('data/y-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    [X, y] = gendata(m, n, d, rho);
    Xt = PCA1(X, d, 0.35);
    csvwrite(Xt_f, Xt(:,1:d));
    csvwrite(y_f, y);
    csvwrite(X_f, X);
    clear all;
end

for rho = 0.01:5:25
    rng(123)
    m = 5000;
    n = 400;
    d = 50;
    rho
    Xt_f = strcat('data/Xt-',int2str(n),'-', int2str(d),'-', num2str(rho),'.csv');
    X_f = strcat('data/X-',int2str(n),'-', int2str(d),'-', num2str(rho),'.csv');
    y_f = strcat('data/y-',int2str(n),'-', int2str(d),'-', num2str(rho),'.csv');
    [X, y] = gendata(m, n, d, rho);
    Xt = PCA1(X, d, 0.35);
    csvwrite(Xt_f, Xt(:,1:d));
    csvwrite(y_f, y);
    csvwrite(X_f, X);
    clear all;
end

