function RFA_plotTree(Tree)
N = size(Tree, 1);
nodes = zeros(1, N);

for i = 1:N
   if Tree(i, 3) ~= 0
       nodes(Tree(i, 3)) = i;
   end
   if Tree(i, 4) ~= 0
       nodes(Tree(i, 4)) = i;
   end    
end

treeplot(nodes);
[x,y] = treelayout(nodes);
strs = cellstr(num2str((1:N)'));
text(x', y', strs, 'VerticalAlignment','bottom','HorizontalAlignment','left', 'FontWeight','bold' )
title({'RFA Tree'},'FontSize',12,'FontName','Times New Roman');
end