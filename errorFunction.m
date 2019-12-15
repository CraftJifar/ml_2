function y = errorFunction(d,y)
e = 0.0;
for i = 1: length(y)
    e = e + (d(i) - y(i)).^2; 
end
y = e ./ 2;
end