% This is ML assignment 1 - simulate single layer NN
% By Jifar Mekonnen , Id: GSR/3602/11

%init the weight matrix by generating random numbers between 0 and 1
fprintf("Iteration No: "+0 +" , weight at initalization stage is: ");
W1 = rand(2,6);
W2 = rand(3,3);

%define the desired vector
desired = [3.4, 9.6, 5.1]';
desired = desired/ norm(desired);

%define the output vector, y
y = [0, 0, 0]';

%define the input vector
x1 = [6.4, 1.3, 2.9, 19.7, 12.5, 2.0, 6.2, 14.9, 5.0, 1.7];
x1 = x1/ norm(x1);
x1p1 = [1, x1(1:5)]';
x1p2 = [1, x1(6:10)]';

% define the acceptable error level , error_min
Error_min = 0.00001;

% define Beta , Beta
Beta = 1;

% define Neu,
Neu = 0.65;

% define vector to store each iteration errors
errors = zeros(1, 2000);

for i = 1 : 2000
    
    % For input X1
   
    % U1
    U1 = W1 * x1p1;
    % V 
    V = sigmoid(U1);
    % U2 
    U2 = W2 * [1, V']';
    % Y
    Y = sigmoid(U2);
    % E
    E1 = desired - Y;
    E1 = E1.^2;
    
    
    % For input X2
    
    % U1
    U1 = W1 * x1p2;
    % V 
    V = sigmoid(U1);
    % U2 
    U2 = W2 * [1, V']';
    % Y
    Y = sigmoid(U2);
    % E
    E2 = desired - Y;
    E2 = E2.^2;
    
    E = 0.5 * sum([E1;E2]);
    
    
    
    errors(i) = E;
    
    plot(1:i,errors(1:i));
    title('Value of error on each iteration');
    xlabel('Iteration');
    ylabel('Error')
    
    
   
    
    if E <= Error_min
        fprintf("Iteration No: "+ i +" , weight after succcessful training is: ");
        W1
        W2
        break;
    end
    
end
if i == 100 && cumulatedError > Error_min
   fprintf("The network is not convergent!");
end


