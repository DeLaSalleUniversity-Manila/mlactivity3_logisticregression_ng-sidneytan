x = load('ml3x.dat');
y = load('ml3y.dat');
plot(x(:,1),y,'o');
xlabel('Exam 1')
ylabel('y')
print('plot1','-dpng');
plot(x(:,2),y,'o');
xlabel('Exam 2')
ylabel('y')
print('plot2','-dpng');
m = 80;
x = [ones(80,1),x];

% pos and neg
pos = find(y == 1);
neg = find(y == 0);
plot(x(pos,2), x(pos,3), '+');
hold on
plot(x(neg,2), x(neg,3), 'o');
xlabel('Exam 1 score')
ylabel('Exam 2 score')
print('plot3','-dpng');


%cost j
theta = zeros(size(x(1,:)))';
max = 7;
j = zeros(max,1);
for num = 1:max
z = x * theta;
g = 1.0 ./ (1.0 + exp(-z));
gradient = (1/m).*x'*(g-y);
H = (1/m).*x'*diag(g)*diag(1-g)*x;
j(num) = (1/m)*sum(-y.*log(g) - (1-y).*log(1-g));
theta = theta - H\ gradient;
end

% boundary line
plot_x = [ min(x(:,2)) - 2,  max(x(:,2)) + 2 ];
plot_y = (-1./theta(3)).*(theta(2).* plot_x + theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off
print('plot4', '-dpng');

%plot J
figure
plot(0:max-1, j, '-');
xlabel('Iteration'); ylabel('J')
print('plot5', '-dpng');

%probability
z = [1 20 80] * theta;
g = 1.0 ./ (1.0 + exp(-z));
prob = 1 - g;

