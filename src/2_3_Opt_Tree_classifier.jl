using DataFrames
using OptimalTrees
using OptImpute
# using DelimitedFiles



function random_splitting(data, split)
           n = nrow(data)
           index = shuffle(1:n)
           train_index = view(index, 1:floor(Int, split*n))
           test_index = view(index, (floor(Int, split*n)+1):n)
           data[train_index,:], data[test_index,:]
       end



# define path to data files
train_filename = "/Users/georgyguryev/Documents/MIT/courses/6.867/2018/Project/Final/full_train_balanced.csv"
test_filename = "/Users/georgyguryev/Documents/MIT/courses/6.867/2018/Project/Final/full_test.csv"


df    = readtable(train_filename , header = true)
test  = readtable(test_filename, header = true)

train, valid = random_splitting(df, 1)


# split data on labels and features
train_X, train_y = convert(Array, train[:,(1:size(train)[2]-1)]),
                     convert(Array, train[:, size(train)[2]])

valid_X, valid_y = convert(Array, valid[:,(1:size(valid)[2]-1)]),
                     convert(Array, valid[:, size(valid)[2]])

test_X, test_y = convert(Array, test[:,(1:size(test)[2]-1)]),
                     convert(Array, test[:, size(test)[2]])


print(unique(train_y))

# perform hyperparameter optimization for batch size
optimal_accuracy = 0
       optimal_minbucket = -1
       optimal_maxdepth = -1
       for i in [1, 5, 7, 10]
           for j in [1, 5, 7, 10]
               model = OptimalTrees.OptimalTreeClassifier(minbucket = i, max_depth = j)
              OptimalTrees.fit!(model, train_X, train_y)
               test_accuracy = mean(OptimalTrees.predict(model, valid_X) .== valid_y)
               if test_accuracy > optimal_accuracy
                   optimal_minbucket = i
                   optimal_maxdepth = j
                   optimal_accuracy = test_accuracy
               end
           end
       end

optimal_minbucket = 10
optimal_maxdepth = 7
#
train_X, train_y = convert(Array, train[:,(1:size(train)[2]-1)]),
                            convert(Array, train[:, size(train)[2]])

print("The best parameters are: minbucket", optimal_minbucket)
print("Max Depth: ", optimal_maxdepth)


# build an Optimal tree and
lnr = OptimalTrees.OptimalTreeClassifier(minbucket = optimal_minbucket, max_depth = optimal_maxdepth, hyperplane_config=[Dict(:sparsity => :all)])
# lnr = OptimalTrees.OptimalTreeClassifier(minbucket = optimal_minbucket, max_depth = optimal_maxdepth)

OptimalTrees.fit!(lnr, train_X, train_y)

# test Optimal tree
print("Training accuracy: ", mean(OptimalTrees.predict(lnr, train_X) .== train_y))
print("Testing accuracy: ", mean(OptimalTrees.predict(lnr, test_X) .== test_y))

predictions = convert(Array,OptimalTrees.predict_proba(lnr, test_X))

writedlm("/Users/georgyguryev/Documents/MIT/courses/6.867/2018/Project/Final/predict.txt", convert(Array,predictions))

score       = OptimalTrees.score(lnr, test_X, test_y, criterion=:entropy)

# print(score)
