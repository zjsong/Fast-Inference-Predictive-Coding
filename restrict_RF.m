function [W_stage, V_stage, weight_localID] = restrict_RF(W_stage, V_stage, nodes, nr_stage, npr_stage, ranges)

p_stage = sqrt(nr_stage);
r = 0;
weight_localID = ones(size(W_stage));
for x = 1 : size(ranges,1)
    for y = 1 : size(ranges,1)
        r = r + 1;
	    region = zeros(p_stage, p_stage);
	    region(ranges(x,:), ranges(y,:)) = 1;
	    row = 0;
	    for c = 1 : p_stage
            for d = 1 : npr_stage
                row = row + 1;
		        regionNodes(row, :) = region(c, :);
            end
        end
        regionNodes = regionNodes(:);
	    for j = nodes(r,:)
            W_stage(j,:) = W_stage(j,:).*regionNodes';
            V_stage(:,j) = V_stage(:,j).*regionNodes;
            weight_localID(j,:) = regionNodes';
        end
        clear regionNodes;
    end
end









