function y = sigmoid(u)
Beta = 1;
y = 1 ./ (1 + exp(-Beta.*u));
end