function [ labels, features ] = gen_random_data( sample_count )

if nargin < 1
    sample_count = 1000;
end

label_count = 4;
feature_dims = 5;
feature_mutrate = zeros(feature_dims, 1) + 0.005;

feature_centers = zeros(feature_dims, label_count);
for i = 1:feature_dims
    feature_centers(i, :) = 1:label_count;
end

labels = zeros(1, sample_count);
features = zeros(feature_dims, sample_count);

for i = 1:sample_count
    new_label = ceil(unifrnd(0, label_count));
    new_feature = feature_centers(:, new_label);
    for j = 1:feature_dims
        if unifrnd(0,1) < feature_mutrate(j)
            new_feature(j) = ceil(unifrnd(0, label_count));
        end
    end
    labels(1, i) = new_label;
    features(:, i) = new_feature;
end

end

