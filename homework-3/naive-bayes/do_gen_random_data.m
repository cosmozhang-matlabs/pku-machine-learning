
[ labels, features ] = gen_random_data;

sample_count = size(labels, 2);
feature_count = size(features, 1);

csvdata = zeros(sample_count, 1 + feature_count);
csvdata(:, 1) = labels';
csvdata(:, 2:end) = features';

diffs = zeros(sample_count, 1);
for i = 1:feature_count
    diffs = diffs + (csvdata(:,1) ~= csvdata(:,1+i));
end

mean(diffs)