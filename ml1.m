% This is ML assignment 1 - simulate single layer NN
% By Jifar Mekonnen , Id: GSR/3602/11

%init the weight matrix by generating random numbers between 0 and 1
fprintf("Iteration No: "+0 +" , weight at initalization stage is: ");
W1 = rand(2,6);
W2 = rand(3,3);

%define the desired vector
desired = [3.4, 9.6, 5.1]';
desired = desired/ norm(desired);


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
    U1X1 = W1 * x1p1;
    % V 
    VX1 = [1;sigmoid(U1X1)];
    % U2 
    U2X1 = W2 * VX1;
    % Y
    Y = sigmoid(U2X1);
    % E
    E1 = desired - Y;
    E1S = E1.^2;
    
    
    % For input X2
    
    % U1
    U1X2 = W1 * x1p2;
    % V 
    VX2 = [1; sigmoid(U1X2)];
    % U2 
    U2X2 = W2 * VX2;
    % Y
    Y = sigmoid(U2X2);
    % E
    E2 = desired - Y;
    E2S = E2.^2;
    
    
    
    % layer 2 learn
    Size_W2 = size(W2);
    Temp_W2 = rand(Size_W2(1), Size_W2(2));
    for p = 1 : Size_W2(1)
        for q =  1: Size_W2(2)
            Temp_W2(p,q) = W2(p,q) - ( Neu .* E1(p) .* sigmoidDervetive(U2X1(p)) * VX1(q));         
        end
    end
    
    % layer 1 learn 
    Size_W1 = size(W1);
    Temp_W1 = rand(Size_W1(1), Size_W1(2));
    for p = 1 : Size_W1(1)
        for q = 1 : Size_W1(2)
            Sum_s = 0;
            for s = 1 : Size_W2(1)
                Sum_s = E1(s) * sigmoidDervetive(U2X1(s)) * W2(s,p);
            end
            Grad = Sum_s * sigmoidDervetive(U1X1(p)) * x1p1(q);
            Temp_W1(p,q) = W1(p,q) - Neu * Grad;
        end
    end
    
    % layer 2 learn
    Size_W2 = size(W2);
    Temp_W2 = rand(Size_W2(1), Size_W2(2));
    for p = 1 : Size_W2(1)
        for q =  1: Size_W2(2)
            Temp_W2(p,q) = W2(p,q) - ( Neu .* E1(p) .* sigmoidDervetive(U2X2(p)) * VX2(q));         
        end
    end
    
    % layer 1 learn 
    Size_W1 = size(W1);
    Temp_W1 = rand(Size_W1(1), Size_W1(2));
    for p = 1 : Size_W1(1)
        for q = 1 : Size_W1(2)
            Sum_s = 0;
            for s = 1 : Size_W2(1)
                Sum_s = E1(s) * sigmoidDervetive(U2X2(s)) * W2(s,p);
            end
            Grad = Sum_s * sigmoidDervetive(U1X2(p)) * x1p2(q);
            Temp_W1(p,q) = W1(p,q) - Neu * Grad;
        end
    end
    
    
    E = 0.5 * sum([E1S;E2S]);
    
    
    errors(i) = E;
    
    plot(1:i,errors(1:i));
    title('Value of error on each iteration');
    xlabel('Iteration');
    ylabel('Error');

    
    if E <= Error_min
        fprintf("Iteration No: "+ i +" , weight after succcessful training is: ");
        W1
        W2
        break;
    end
    
    
    
    W1 = Temp_W1
    W2 = Temp_W2
    
end



