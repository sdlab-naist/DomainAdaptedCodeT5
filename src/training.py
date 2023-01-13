
# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from tqdm import tqdm


import wandb


# define a rich console logger
console = Console(record=True)

def zip_with_preprocess(masked_code,mask):
    masked_code = list(map(lambda x:x.replace("<x>","<extra_id_0>").replace("\n",""),masked_code))
    mask = list(map(lambda x:x.replace("<z>","").replace("\n",""),mask))
    return zip(masked_code,mask)

def load_dataset(path):
    masked_code_lines = []
    mask_lines = []
    with open(path) as masked_code:
        masked_code_lines = masked_code.readlines()
    with open(path.replace("masked_code","mask")) as mask:
        mask_lines = mask.readlines()
    df_list = [[masked_code,mask] for masked_code,mask in zip_with_preprocess(masked_code_lines,mask_lines)]
    df = pd.DataFrame(df_list,columns=["text","headlines"])
    return df

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def train(epoch, tokenizer, model, device, loader, optimizer,accumulation_steps):

    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    train_iterator = tqdm(enumerate(loader, 0),total=len(loader))
    for _, data in train_iterator:
        optimizer.zero_grad()
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0].mean()
        loss = loss / accumulation_steps
        loss.backward()

        if (_ + 1) % accumulation_steps ==0 or (_ + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        wandb.log({"train_batch_loss": loss.item()})
    


def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():

      train_iterator = tqdm(enumerate(loader, 0),total=len(loader))
      for _, data in train_iterator:
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.module.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          
          predictions.extend(preds)
          actuals.extend(target)

  return predictions, actuals


def T5Trainer(
    dataframe,eval_dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """
    
    os.makedirs(output_dir, exist_ok=True)

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = RobertaTokenizer.from_pretrained(model_params["TOKENIZER"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])


    if device == 'cuda':
        model = torch.nn.DataParallel(model) # make parallel
    model = model.to(device)
    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_dataset = dataframe
    val_dataset = eval_dataframe
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": int(model_params["TRAIN_BATCH_SIZE"]/model_params["GRADIENT_ACCUMULATION"]),
        "shuffle": True,
        "num_workers": 4,
        "pin_memory":True
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 4,
        "pin_memory":True
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")
    
    
    max_ppr = 0
    early_stopping = int(model_params["EARLY_STOPPING"]),
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer,model_params["GRADIENT_ACCUMULATION"])

        if (epoch + 1) % model_params["EPOCHS_CHECK_POINT"] == 0:
            path = os.path.join(output_dir, "check_point_"+str(epoch+1))
            model.module.save_pretrained(path)

        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        df["Perfect Prediction"] = df["Generated Text"] == df["Actual Text"]

        
        ppr = (df["Perfect Prediction"]==True).sum()/len(df)
        wandb.log({"epoch":epoch,"perfect_predictions":ppr})
        if ppr > max_ppr :
            max_ppr = ppr
            early_stopping = int(model_params["EARLY_STOPPING"])
            path = os.path.join(output_dir, "highest_model")
            model.module.save_pretrained(path)
        else:
            early_stopping -= 1
            if early_stopping == 0:
                console.log(f"Early stopped in {epoch+1} epochs !!\n")
                break

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.module.save_pretrained(path)
    tokenizer.save_pretrained(path)

    
    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))
    
    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")



# let's define model parameters specific to T5
model_params = {
}



import typer


def main(
    model: str = typer.Option("Salesforce/codet5-small", help="model name or path"),
    batch: int = typer.Option(1024, help="train batch size"),
    gradient_accumulation: int = typer.Option(8, help="gradient accumulation step"),
    early_stopping:int = typer.Option(2, help="If the accuracy doesn't improve continuously, stop."),
    output:str = typer.Option(".", help="output directory"),
    data:str = typer.Option(...),
):

    model_params = {
        "MODEL": model,  # model_type
        "TOKENIZER":"Salesforce/codet5-small",
        "TRAIN_BATCH_SIZE": batch,  # training batch size
        "GRADIENT_ACCUMULATION":gradient_accumulation, # the number of steps for gradient accumulation
        "VALID_BATCH_SIZE": int(batch/gradient_accumulation),  # validation batch size
        "TRAIN_EPOCHS": 50,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
        "SEED": 40,  # set seed for reproducibility
        "EARLY_STOPPING":early_stopping, # if the model does not upgrade its accuracy in the validation step for EARLY_STOPPING epochs, training will be stopped.
        "EPOCHS_CHECK_POINT":1000, # the model is saved as checkpoint for each 10 epochs.
        "DATA_PATH":data,
        "OUTPUT":output
    }

    train_df = load_dataset(os.path.join(model_params["DATA_PATH"],"training_masked_code.txt"))
    eval_df = load_dataset(os.path.join(model_params["DATA_PATH"],"evalation_masked_code.txt"))

    wandb.login()
    run =wandb.init(config=model_params)

    T5Trainer(
        dataframe=train_df,
        eval_dataframe=eval_df,
        source_text="text",
        target_text="headlines",
        model_params=model_params,
        output_dir= os.path.join(model_params["OUTPUT"],run.name),
    )


if __name__ == "__main__":
    typer.run(main)
