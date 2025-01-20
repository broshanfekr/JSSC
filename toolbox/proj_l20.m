function Z_proj = proj_l20(Z, k)
    % PROJ_L20 Projects a matrix Z onto the L2,0 norm space with k non-zero rows.
    %
    % Inputs:
    %   Z - Input matrix (m x n)
    %   k - Number of rows to keep (based on their L2 norm)
    %
    % Output:
    %   Z_proj - Matrix after L2,0 projection (m x n)

    % Step 1: Compute L2 norm of each row
    row_norms = sqrt(sum(Z.^2, 2));
    
    % Step 2: Sort the row norms in descending order
    [~, sorted_indices] = sort(row_norms, 'descend');
    
    % Step 3: Retain only the top-k rows, set others to zero
    Z_proj = zeros(size(Z));
    top_k_indices = sorted_indices(1:k);  % Indices of top-k rows
    Z_proj(top_k_indices, :) = Z(top_k_indices, :);
end