function y = sigmoidDervetive(u)
Beta = 1;
y = Beta .* exp(-Beta.* u) ./ (1 + exp(-Beta.*u)) .^ 2;
end