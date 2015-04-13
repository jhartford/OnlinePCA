%clear all;
%X = csvread('mnistHelper/mnist.csv');
%y = csvread('mnistHelper/mnist-label.csv');

d = 20;
Xt = PCA1(X, d, 0.35);