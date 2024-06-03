import os
import torch
import datetime
import numpy as np
from typing import Union, Callable
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

class Training:
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Union[None, torch.optim.lr_scheduler.LRScheduler], 
        ckpt_path: str,  # path to save losses and state dicts
    ):
        self.losses = {"train_loss": [], "evaluation": {}}
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        assert ckpt_path.endswith(".pth"), ckpt_path
        if os.path.exists(ckpt_path):
            self.resume_from_checkpoint(ckpt_path)
        else:
            print("start training from scratch")
        self.ckpt_path = ckpt_path
    
    def resume_from_checkpoint(self, path: str):
        """Load losses and state dicts from the given path."""
        assert path is not None and path.endswith(".pth"), path
        
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt["losses"].keys() == self.losses.keys(), (ckpt["losses"].keys(), self.losses.keys())
        
        self.losses = ckpt["losses"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.lr_scheduler is not None:
            assert "lr_scheduler" in ckpt, ckpt.keys()
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        
        print(f"resume training from {path}")
    
    def save_checkpoint(self):
        """Save losses and state dicts to self.ckpt_path."""
        ckpt = {"losses": self.losses, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, self.ckpt_path)
        print(f"checkpoint saved at {self.ckpt_path}")
    
    def plot_loss(self):
        y = np.array(self.losses["train_loss"])
        plt.plot(np.log(y), label="train loss")
        
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.legend()
        
        plt.show()
        plt.close()
    
    def run(
        self, 
        max_niters: int,  # maximum number of iterations to train the model
        train_loader: torch.utils.data.DataLoader, 
        train_iteration: Callable,
        evaluate: Union[Callable, None] = None, 
        test: Union[Callable, None] = None,
        plot_loss: Union[Callable, None] = None,
        gradient_accumulation_step: int = 1,
        print_step: int = 1, 
        plot_step: Union[int, None] = 100,
        save_step: Union[int, None] = 100,
        eval_step: Union[int, None] = None,
        test_step: Union[int, None] = 100,  # if not None, `test(self.model)` will be called every `test_step` training iterations
    ):
        if len(self.losses["train_loss"]) >= max_niters:
            return
        
        # training loop
        start_time = datetime.datetime.now()
        for ep in range(int(np.ceil((max_niters - len(self.losses["train_loss"])) / len(train_loader)))):
            for ite, batch in enumerate(train_loader):
                # training iteration
                loss = train_iteration(self.model, batch)
                assert torch.isnan(loss) == False
                
                self.losses["train_loss"].append(loss.item())
                
                loss = loss / gradient_accumulation_step
                loss.backward()
                
                niters = len(self.losses["train_loss"])
                if niters % gradient_accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # print training iteration
                if ep == ite == 0 or niters % print_step == 0:
                    text = [
                        f"ite {niters - 1} / {max_niters}:", 
                        f"loss={self.losses['train_loss'][-1]:.6f}",
                    ]
                    if torch.backends.mps.is_available():
                        text += [
                            f"allocated={torch.mps.driver_allocated_memory() * 1e-9:.2f}G",
                        ]
                    text += [
                        f"elapsed={datetime.datetime.now() - start_time}",
                    ]
                    print(" ".join(text))
                
                # export losses, model, optimizer, lr_scheduler
                if save_step is not None and niters % save_step == 0:
                    self.save_checkpoint()
                
                # evaluate the model on a test set
                if eval_step is not None and evaluate is not None and niters % eval_step == 0:
                    print("evaluation...")
                    scores = evaluate(self.model)
                    self.losses["evaluation"][niters - 1] = scores
                    print(f"{scores} elapsed={datetime.datetime.now() - start_time}")
                
                # plot losses
                if plot_step is not None and niters % plot_step == 0:
                    if plot_loss is not None:
                        plot_loss(self.losses)
                    else:
                        self.plot_loss()
                
                # test the model as defined in the function
                if test_step is not None and test is not None and niters % test_step == 0:
                    print("test...")
                    test(self.model)
                    print(f"elapsed={datetime.datetime.now() - start_time}")
                
                # termination
                if niters >= max_niters:
                    delta = datetime.datetime.now() - start_time
                    print(f"training finished in {delta}")
                    return
