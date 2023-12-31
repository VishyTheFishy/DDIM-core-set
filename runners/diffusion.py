import os
import logging
import time
import glob
import matplotlib.pyplot as plt
import csv
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from k_means import KMeans
import pandas as pd
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import statistics


from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        

        
    def train(self):
        coreset_method = "k_means"
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        model = Model(config)
        test_loader = data.DataLoader(test_dataset,batch_size=config.training.batch_size,shuffle=True,num_workers=config.data.num_workers,)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        test_losses = []
        steps = []
        for epoch in range(start_epoch, 400):#self.config.training.n_epochs):
            total = 0
            for i, (x, y) in enumerate(test_loader):
                print(epoch)
                n = x.size(0)
                model.train()
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b).item()
                total += loss
                print(total)
            test_losses.append(total)
            steps.append(step)
                

            train_loader = data.DataLoader(dataset,batch_size=config.training.batch_size,shuffle=True,num_workers=config.data.num_workers,)

            data_start = time.time()
            data_time = 0
            print(len(dataset))
            for i, (x, y) in enumerate(train_loader):
                print(epoch)
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
            score_loader = data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config.data.num_workers)
            if(coreset_method == "k_means" and epoch == 2):
                select = "distance"
                m = 250
                print(model)
                kmeans = MiniBatchKMeans(n_clusters=100,
                    random_state=0,
                    batch_size=128,
                    n_init="auto")
                for i, (x, y) in enumerate(train_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    t = torch.ones(size=(1,)).type(torch.LongTensor).to(self.device)*500
                    embedding = model(x,t,True)
                    kmeans = kmeans.partial_fit(embedding)

                    
                clusters_distance = [[] for _ in range(100)]
                best_clusters = [[] for _ in range(100)]

                clusters_losses = [[] for _ in range(100)]
                cluster_count = np.zeros(100)
                
                for i, (x, y) in enumerate(score_loader):
                    print(i)
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    t = torch.ones(size=(1,)).type(torch.LongTensor).to(self.device)*500
                    e = torch.randn_like(x)
                    b = self.betas

                    embedding = model(x,t,True)
                    cluster = kmeans.predict(embedding)[0]
                    cluster_count[cluster] += 1
                    if(select == "distance"):
                        distance = kmeans.transform(embedding)[0][cluster]
                        
                        for j, current in enumerate(clusters_distance[cluster]):
                            if(current > distance):
                                clusters_distance[cluster].insert(j,distance)
                                best_clusters[cluster].insert(j,i)
                                break
                        else:
                            clusters_distance[cluster].append(distance)
                            best_clusters[cluster].append(i)
                    if(select == "loss"):
                        loss = loss_registry[config.model.type](model, x, t, e, b).item()
                        for j, current in enumerate(clusters_losses[cluster]):
                            if(current > loss):
                                best_clusters[cluster].insert(j,i)
                                clusters_losses[cluster].insert(j,loss)
                                break
                        else:
                            best_clusters[cluster].append(i)
                            clusters_losses[cluster].append(loss)

                
                    
                coreset = []
                for cluster in best_clusters:
                    for i, point in enumerate(cluster):
                        if(i == 50):
                            break
                        coreset.append(point)
                dataset = torch.utils.data.Subset(dataset, coreset)
            
                
            if(coreset_method == "loss" and len(dataset) > 25000):
                scores = []
                coreset = []
                
                for i, (x, y) in enumerate(score_loader):
                    if(i == 1000):
                        break
                    n = x.size(0)
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    e = torch.randn_like(x)
                    b = self.betas
                        # antithetic sampling
                    
                    t = torch.ones(size=(1,)).type(torch.LongTensor).to(self.device)*20#config.select_t
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    loss = loss_registry[config.model.type](model, x, t, e, b)
                    scores.append(loss.item())
                threshold = statistics.mean(scores)-statistics.stdev(scores)
                for i, (x, y) in enumerate(score_loader):
                    n = x.size(0)
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    e = torch.randn_like(x)
                    b = self.betas
                        # antithetic sampling
                    t = torch.ones(size=(1,)).type(torch.LongTensor).to(self.device)*20#config.select_t
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    loss = loss_registry[config.model.type](model, x, t, e, b)
                    if(loss.item() > threshold):
                        coreset.append(i)
                    print(loss.item())
                dataset = torch.utils.data.Subset(dataset, coreset)
                print(len(dataset))
                plt.hist(scores,bins=50)
                plt.savefig("hist_"+str(epoch)+".png")

            
        f = open('losses.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(steps)
        writer.writerow(test_losses)
        
        plt.close()
        plt.plot(steps,test_losses)
        plt.yscale('log')
        plt.savefig("losscurve.png")


    def sample_(self):
        
        args, config = self.args,self.config
        print(config)
        model = Model(self.config)
        
        name = "cifar10"
        ckpt = get_ckpt_path(f"ema_{name}")
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        print(model)
        
        scores = []
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config.data.num_workers)
        for i, (x, y) in enumerate(train_loader):
            if(i == 10000):
                break
            print(torch.cuda.memory_allocated(0),i)

            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas
                # antithetic sampling
            t = torch.ones(size=(1,)).type(torch.LongTensor).to(self.device)*100#config.select_t
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
               
            
            """loss_z = torch.clone(loss_pos).detach()
            loss_z.backward(torch.ones(x.size()).to(self.device))         
            grad = inputs.grad.data + 0.0
            norm_grad = grad.norm().item()
            z = torch.sign(grad).detach() + 0.
            z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)"""

            
            """z = torch.ones(x.size()).to(self.device)*.0001
            d = torch.ones(x.size()).to(self.device)*.00001
            
            loss_orig = loss_registry[config.model.type](model, x + z, t, e, b)
            loss_orig_d = loss_registry[config.model.type](model, x + z + d, t, e, b)
            grad_orig = (loss_orig_d - loss_orig)/torch.linalg.norm(d)

            loss_pos = loss_registry[config.model.type](model, x - z, t, e, b)
            loss_pos_d = loss_registry[config.model.type](model, x - z + d, t, e, b)
            grad_pos = (loss_pos_d - loss_pos)/torch.linalg.norm(d)

            grad_diff = (grad_orig - grad_pos)/(2*torch.linalg.norm(z))
            print(grad_diff)
            
            
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), x, )[0]"""

            loss = loss_registry[config.model.type](model, x, t, e, b)
            
            scores.append(loss.item())
        print(scores)
        plt.hist(scores,bins=200)
        plt.savefig("hist1.png")

            

        

        
        
    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
