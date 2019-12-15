function y = findWeightedSum(w, x) 
y = 0.0;
for i = 1: length(x)
    y = y + (w(i).*x(i));
end
end