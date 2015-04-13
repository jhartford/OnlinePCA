%clear all;
%X = csvread('mnistHelper/mnist.csv');
%y = csvread('mnistHelper/mnist-label.csv');
% 
% d = 50;
% Xt = PCA1(X, d, 0.35);
% csvwrite('mnist_OPCA.csv', Xt(:,1:d));

clear all;
% 
% for n = 100:200:1000
%     tic
%     rng(123)
%     m = 5000;
%     d = 50;
%     rho = 5;
%     n
%     Xt_f = strcat('data/Xt-',int2str(n),'-', int2str(d),'-', num2str(rho),'-', num2str(0.25),'.csv');
%     X_f = strcat('data/X-',int2str(n),'-', int2str(d),'-', num2str(rho),'-', num2str(0.25),'.csv');
%     y_f = strcat('data/y-',int2str(n),'-', int2str(d),'-', num2str(rho),'-', num2str(0.25),'.csv');
%     [X, y] = gendata(m, n, d, rho, 1);
%     Xt = PCA1(X, d, 0.25);
%     csvwrite(Xt_f, Xt(:,1:d));
%     csvwrite(y_f, y);
%     csvwrite(X_f, X);
%     clear all;
%     toc
% end


for n = 100:200:1000
    tic
    rng(123)
    m = 5000;
    d = 50;
    rho = 5;
    n
    Xt_f = strcat('data/l2Xt-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    X_f = strcat('data/l2X-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    y_f = strcat('data/l2y-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    [X, y] = gendata(m, n, d, rho, 2);
    Xt = PCA1(X, d, 0.35);
    csvwrite(Xt_f, Xt(:,1:d));
    csvwrite(y_f, y);
    csvwrite(X_f, X);
    clear all;
    toc
end


for d = 20:20:100
    rng(123)
    m = 5000;
    n = 400;
    rho = 5;
    d
    Xt_f = strcat('data/l2Xt-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    X_f = strcat('data/l2X-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    y_f = strcat('data/l2y-',int2str(n),'-', int2str(d),'-', int2str(rho),'.csv');
    [X, y] = gendata(m, n, d, rho, 2);
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
    Xt_f = strcat('data/l2Xt-',int2str(n),'-', int2str(d),'-', num2str(rho),'.csv');
    X_f = strcat('data/l2X-',int2str(n),'-', int2str(d),'-', num2str(rho),'.csv');
    y_f = strcat('data/l2y-',int2str(n),'-', int2str(d),'-', num2str(rho),'.csv');
    [X, y] = gendata(m, n, d, rho, 2);
    Xt = PCA1(X, d, 0.35);
    csvwrite(Xt_f, Xt(:,1:d));
    csvwrite(y_f, y);
    csvwrite(X_f, X);
    clear all;
end

