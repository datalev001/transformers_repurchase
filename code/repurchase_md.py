import pandas as pd
from sklearn.model_selection import train_test_split
from lifetimes import BetaGeoFitter
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp

tran_df = pd.read_excel('online_retail_II.xlsx')
# define rules and clean data 1:
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity']>0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]

# define rules and clean data 2:
grp = ['Invoice', 'StockCode','Description', 'Quantity', 'InvoiceDate']
tran_df = tran_df.drop_duplicates(grp)
tran_df['transaction_date'] = tran_df['InvoiceDate'].dt.date
tran_df = tran_df.groupby(['Customer ID', 'Description','transaction_date'])['Quantity'].sum().reset_index()

# get cleanned trasaction data
tran_df_sel = tran_df.copy()
tran_df_sel['trans_date'] = pd.to_datetime(tran_df_sel['transaction_date'], format = '%Y-%m-%d')

summary_data_df = summary_data_from_transaction_data\
    (tran_df_sel, 'Customer ID', 
    'trans_date', 
    observation_period_end='2010-11-10').reset_index()

df2 = tran_df_sel.groupby('Customer ID').agg(
    avg_Quantity=('Quantity', 'mean'),  # Average quantity per customer
    uniq_product_n=('Description', pd.Series.nunique)  # Count of unique products
).reset_index()

# target DF 
target_df = (tran_df_sel.query("trans_date > '2010-11-10' and Quantity > 10")\
 .groupby('Customer ID')['Quantity'].sum().reset_index().assign(target=1)[['Customer ID', 'target']])


## binning function
def bin_and_format(column, bins):
    # Try to create the specified number of bins
    try:
        categorized = pd.qcut(column, q=bins, duplicates='drop')
        # Successfully created bins, now assign numeric labels for simplicity
        unique_bins = len(categorized.cat.categories)
        labels = [i for i in range(1, unique_bins+1)]
        categorized = pd.qcut(column, q=bins, labels=labels, duplicates='drop')
        return categorized
    except ValueError as e:
        # If ValueError occurs, print the error and attempt to reduce bins
        print(f"Error: {e}")
        unique_bins = len(column.unique())
        try:
            labels = [i for i in range(1, unique_bins)]
            categorized = pd.qcut(column, q=unique_bins, labels=labels, duplicates='drop')
            return categorized
        except Exception as e:
            # If it still fails, print the final error and return None or original column
            print(f"Final Error: {e}")
            return column  # or return None if you want to indicate failure

summary_data_df = pd.merge(summary_data_df, df2, on ='Customer ID',  how = 'left')
summary_data_df = summary_data_df.fillna(0)

# Applying the function to the 'frequency', 'recency', and 'T' columns
summary_data_df['frequency'] = bin_and_format(summary_data_df['frequency'], 10).astype(str)
summary_data_df['recency'] = bin_and_format(summary_data_df['recency'], 10).astype(str)
summary_data_df['T'] = bin_and_format(summary_data_df['T'], 10).astype(str)
summary_data_df['avg_Quantity'] = bin_and_format(summary_data_df['avg_Quantity'], 10).astype(str)
summary_data_df['uniq_product_n'] = bin_and_format(summary_data_df['uniq_product_n'], 10).astype(str)

summary_data_df['sum_text'] = '[' + summary_data_df['frequency'] +\
    ',' + summary_data_df['recency'] + ',' +\
      summary_data_df['T'] + ',' + summary_data_df['avg_Quantity'] + \
      ',' + summary_data_df['uniq_product_n'] + ']'

# Selecting the required columns to display
summary_data_df = summary_data_df[['Customer ID', 'sum_text']]

# get most freq purchased product for each customer
transaction_counts = tran_df_sel.groupby(['Customer ID', 'Description']).size().reset_index(name='counts')
idx = transaction_counts.groupby(['Customer ID'])['counts'].transform(max) == transaction_counts['counts']
max_transactions = transaction_counts[idx]
max_transactions = max_transactions.drop_duplicates(subset=['Customer ID'], keep='first')
max_transactions_df = max_transactions[['Customer ID', 'Description']]
max_transactions_df.columns = ['Customer ID', 'Description_max']

# obtain modeling data
summary_data_df = pd.merge(summary_data_df, target_df, on ='Customer ID',  how = 'left')
summary_data_df = pd.merge(summary_data_df, max_transactions_df, on ='Customer ID',  how = 'left')
summary_data_df['sum_text'] = summary_data_df['sum_text'] + summary_data_df['Description_max']
summary_data_df = summary_data_df.fillna(0)


# lifetimes method: BG/NBD model
# Normally, you'd prepare RFM (recency, frequency, monetary) features for this model

df1 = summary_data_from_transaction_data(
    tran_df_sel, 'Customer ID', 
    'trans_date',  observation_period_end='2010-11-10').reset_index()

target_df = (tran_df_sel.query("trans_date > '2010-11-10' and Quantity > 10")
             .groupby('Customer ID')['Quantity'].sum().reset_index().assign(target=1)
             [['Customer ID', 'target']])

df1 = pd.merge(df1, target_df, on='Customer ID', how='left')
df1 = df1.fillna(0)

ks_sum = 0 
auc_sum = 0
acc_sum = 0
TTT = 0
for  jjj in range(50):
    # Split data into train and test sets
    train_df, test_df, train_labels, test_labels = train_test_split(df1, df1['target'], test_size=0.25)
    
    # Fitting the BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.5)
    bgf.fit(train_df['frequency'], train_df['recency'], train_df['T'])
    
    # Predicting probabilities of transactions
    test_df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        1, # assuming we're looking 1 unit time into the future
        test_df['frequency'], 
        test_df['recency'], 
        test_df['T']
    )
    
    # Calculating AUC
    auc_score = roc_auc_score(test_df['target'], test_df['predicted_purchases'])
        
    # Calculating KS Statistic
    fpr, tpr, thresholds = roc_curve(test_df['target'], test_df['predicted_purchases'])
    ks_statistic = max(tpr - fpr)
    
    # Alternative method to calculate KS Statistic using scipy
    ks_statistic, ks_pvalue = ks_2samp(
        test_df[test_df['target'] == 1]['predicted_purchases'],
        test_df[test_df['target'] == 0]['predicted_purchases']
    )
    
    accuracy1 = accuracy_score(test_df['target'].astype(int), (test_df['predicted_purchases']>test_df['predicted_purchases'].mean()) + 0)
    accuracy2 = accuracy_score(test_df['target'].astype(int), (test_df['predicted_purchases']>0.5) + 0)
    
    ks_sum = ks_sum + ks_statistic
    auc_sum = auc_sum + auc_score
    acc_sum = acc_sum + accuracy2
    TTT = TTT + 1
    print (jjj)
    
print("auc: {:.3f}".format(auc_sum/TTT))
print("KS Statistic (scipy): {:.3f}".format(ks_sum/TTT))
print("accuracy: {:.3f}".format(acc_sum/TTT))


#######transformer: BERT model#######
df = summary_data_df.copy()
df['description'] = df['sum_text']
df.target.mean()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize descriptions
df['input_ids'] = df['sum_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512))

# Padding
max_len = max(df['input_ids'].apply(len))
df['input_ids'] = df['input_ids'].apply(lambda x: x + [0]*(max_len-len(x)))

# Dataset class
class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {'input_ids': encodings}
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['input_ids'].tolist(), df['target'].tolist(), test_size=0.25)

# Create torch dataset
train_dataset = CustomerDataset(train_texts, train_labels)
test_dataset = CustomerDataset(test_texts, test_labels)

# Load a pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer with possible debugging
def forward_pass_with_logging(model, inputs):
    outputs = model(**inputs)
    print(f"Output size: {outputs.logits.size()}")  # Log output size
    print(f"Labels size: {inputs['labels'].size()}")  # Log label size
    return outputs

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {'dummy_metric': 0}  
)

# Run a simple test before full training to check for size issues
batch = next(iter(DataLoader(train_dataset, batch_size=2)))
forward_pass_with_logging(model, batch)

# Train the model
trainer.train()

# Prediction and Evaluation
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Calculate metrics
auc = roc_auc_score(test_labels, preds)
accuracy = accuracy_score(test_labels, preds)

# Calculate the KS statistic
fpr, tpr, thresholds = roc_curve(test_labels, preds)
ks_statistic = np.max(tpr - fpr)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"KS Statistic: {ks_statistic}")


######transformer: distilbert model#################
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import numpy as np

# Load and prepare the dataset
df = summary_data_df.copy()
df['description'] = df['sum_text']

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize descriptions
df['input_ids'] = df['sum_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512))

# Padding
max_len = max(df['input_ids'].apply(len))
df['input_ids'] = df['input_ids'].apply(lambda x: x + [0]*(max_len - len(x)))

# Dataset class
class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {'input_ids': encodings}
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['input_ids'].tolist(), df['target'].tolist(), test_size=0.25)

# Create torch dataset
train_dataset = CustomerDataset(train_texts, train_labels)
test_dataset = CustomerDataset(test_texts, test_labels)

# Load a pre-trained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=4, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=16, 
    warmup_steps=500, 
    weight_decay=0.01, 
    logging_dir='./logs', 
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Prediction and Evaluation
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Calculate metrics
auc = roc_auc_score(test_labels, preds)
accuracy = accuracy_score(test_labels, preds)
fpr, tpr, thresholds = roc_curve(test_labels, preds)
ks_statistic = np.max(tpr - fpr)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"KS Statistic: {ks_statistic}")

#############GPT2############
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import numpy as np

# Load and prepare the dataset
df = summary_data_df.copy()
df['description'] = df['sum_text'].fillna(" ")  # Ensuring no None values

# Initialize the tokenizer and explicitly set a padding token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 doesn't have a PAD token by default, setting it to the EOS token (you can choose any token)
pad_token = tokenizer.eos_token
tokenizer.pad_token = pad_token

# Tokenize and pad
df['input_ids'] = df['description'].apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)
)
max_len = max(len(ids) for ids in df['input_ids'])  # Calculate max length for padding
df['input_ids'] = df['input_ids'].apply(
    lambda ids: ids + [tokenizer.pad_token_id] * (max_len - len(ids))  # Apply padding
)

# Verify pad token is set and recognized in tokenizer
if tokenizer.pad_token is None:
    raise AssertionError("Padding token not set in tokenizer.")

# Dataset class
class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {'input_ids': encodings}
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# Prepare datasets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['input_ids'].tolist(), df['target'].tolist(), test_size=0.25
)
train_dataset = CustomerDataset(train_texts, train_labels)
test_dataset = CustomerDataset(test_texts, test_labels)

# Load and configure the GPT-2 model
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id  # Ensure the model recognizes the pad token

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Execute training
trainer.train()

# Evaluation
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
auc = roc_auc_score(test_labels, preds)
accuracy = accuracy_score(test_labels, preds)
fpr, tpr, thresholds = roc_curve(test_labels, preds)
ks_statistic = np.max(tpr - fpr)

print(f"AUC: {auc}, Accuracy: {accuracy}, KS Statistic: {ks_statistic}")


###########transformer: Albert model##########
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import numpy as np

# Copying the dataframe
df = summary_data_df.copy()
df['description'] = df['sum_text']
print(df.target.mean())

# Initialize the tokenizer for ALBERT
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Tokenize descriptions
df['input_ids'] = df['sum_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512))

# Padding
max_len = max(df['input_ids'].apply(len))
df['input_ids'] = df['input_ids'].apply(lambda x: x + [tokenizer.pad_token_id] * (max_len - len(x)))

# Dataset class
class CustomerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {'input_ids': encodings}
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['input_ids'].tolist(), df['target'].tolist(), test_size=0.25)

# Create torch datasets
train_dataset = CustomerDataset(train_texts, train_labels)
test_dataset = CustomerDataset(test_texts, test_labels)

# Load a pre-trained ALBERT model for sequence classification
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer with possible debugging
def forward_pass_with_logging(model, inputs):
    outputs = model(**inputs)
    print(f"Output size: {outputs.logits.size()}")  # Log output size
    print(f"Labels size: {inputs['labels'].size()}")  # Log label size
    return outputs

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {'dummy_metric': 0}  
)

# Run a simple test before full training to check for size issues
batch = next(iter(DataLoader(train_dataset, batch_size=2)))
forward_pass_with_logging(model, batch)

# Train the model
trainer.train()

# Prediction and Evaluation
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Calculate metrics
auc = roc_auc_score(test_labels, preds)
accuracy = accuracy_score(test_labels, preds)
fpr, tpr, thresholds = roc_curve(test_labels, preds)
ks_statistic = np.max(tpr - fpr)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"KS Statistic: {ks_statistic}")



