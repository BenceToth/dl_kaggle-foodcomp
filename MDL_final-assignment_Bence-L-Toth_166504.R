
# Kaggle competition
## https://www.kaggle.com/c/ifood-2019-fgvc6/overview

# splitting large .tar files to upload to AWS EC2 Rstudio client
## https://www.tecmint.com/split-large-tar-into-multiple-files-of-certain-size/

# Extracting images from .tar files
untar('raw/val.tar', exdir = 'data/')
untar('raw/test.tar', exdir = 'data/')
untar('raw/backup.tar.joined', exdir = 'data/')

# set up folders
train_dir <- file.path('data/train_set')
val_dir <- file.path('data/val_set')
test_dir <- file.path('data/test_set')

# get classes
class_list <- read.delim('raw/class_list.txt', sep = ' ', header = F)
names(class_list) <- c('id', 'class_name')

# get labels
train_labels <-  read.csv('raw/train_info.csv', header = F)
val_labels <-  read.csv('raw/val_info.csv', header = F)

# create folder structure: 1 folder for each class
lapply(paste0(train_dir,'/', 0:250), dir.create)
lapply(paste0(val_dir,'/', 0:250), dir.create)

# move images to their class folder
for (i in 1:length(train_labels[, 1])) {
  file.copy(file.path(train_dir, train_labels[i, 1]),
            file.path(train_dir, train_labels[i, 2]))
}

for (i in 1:length(val_labels[, 1])) {
  file.copy(file.path(val_dir, val_labels[i, 1]),
            file.path(val_dir, val_labels[i, 2]))
}

library(keras)

# image size to scale down to
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# define batch size and number of epochs
batch_size <- 150
epochs <- 10


model <- keras_model_sequential() %>% 
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(251) %>% 
  layer_activation("softmax")

train_datagen = image_data_generator(
  rescale = 1/255
)

# Get images
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = target_size,  
  classes = as.character(class_list$id),
  class_mode = "categorical",
  seed = 42
)

validation_generator <- flow_images_from_directory(
  val_dir,
  test_datagen,
  target_size = target_size,
  classes = as.character(class_list$id),
  class_mode = "categorical",
  seed = 42
)


# Compile model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Fit model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(118475/batch_size),
  epochs = epochs,
  validation_data = validation_generator,
  validation_steps = as.integer(11994/batch_size),
  callbacks = callback_early_stopping(monitor = 'val_loss', patience = 5)
)

# Create test folder structure
dir.create(file.path(test_dir, 'test'))

test_names <- read.csv('raw/test_info.csv', header = F)


for (i in 1:length(test_names[, 1])) {
  file.copy(file.path(test_dir, test_names[i, 1]),
            file.path(test_dir, 'test/'))
}

# Predict on test set
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = target_size,
  batch_size = 1,
  class_mode = NULL
)

predictions <- model %>% 
  predict_generator(test_generator, steps = 28377)


# format output file

row.names(predictions) <- test_names[, 1]
colnames(predictions) <- class_list[, 1]

library(dplyr)
library(tidyverse)


predictions <- predictions %>% 
  as.data.frame() %>% 
  rownames_to_column('img_name') %>% 
  gather(key = "class",
         value = "confidence",
         -img_name)

predictions <- predictions %>% 
  group_by(img_name) %>% 
  arrange(desc(confidence)) %>% 
  top_n(3) %>% 
  ungroup() %>% 
  select(-confidence) %>% 
  arrange(img_name) 

predictions <- predictions %>% 
  group_by(img_name) %>% 
  mutate(pred = 1:3) %>% 
  ungroup() %>% 
  spread(pred, class) 

food_predictions <- predictions %>% 
  unite('label', `1`, `2`, `3`, sep = ' ')


write.csv(food_predictions, 'food_predictions.csv', row.names = F)

