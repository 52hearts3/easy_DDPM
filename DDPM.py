import torch
import math
from  torch import nn,optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader


#它通过逐步添加噪声来破坏图像，然后通过反向过程逐步去噪来恢复图像
#位置嵌入在这个过程中非常重要，因为它帮助模型在每个时间步正确地处理和去噪数据。
class PositionEmbedding(nn.Module):  #[x,dim]==>[x,dim]
    def __init__(self,dim,scale):
        super(PositionEmbedding,self).__init__()

        self.half_dim=dim//2  #确保 dim 是偶数，因为我们需要将其分成两部分来计算正弦和余弦。
        self.scale=scale

    def forward(self,x):
        device=x.device
        half_dim=self.half_dim
        w_k=math.log(10000.0)/half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -w_k)
        emb=torch.outer(x*self.scale,emb) #计算输入时间步与指数衰减向量的外积,这个张量表示时间步的线性变换
        # 得到一个形状为 (N, half_dim) 的张量。x.T(N,1)*emb(1,half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) #计算正弦和余弦，并将它们沿最后一个维度拼接，得到一个形状为 (N,dim) 的张量
        return emb
test=torch.randn(10)
model=PositionEmbedding(dim=32,scale=1.0)
print(model(test).size())

class AttentionBlock(nn.Module):
    def __init__(self,ch_in):
        super(AttentionBlock,self).__init__()

        self.ch_in=ch_in
        self.to_qkv=nn.Conv2d(ch_in,ch_in*3,kernel_size=1,stride=1,padding=0)
        self.to_out=nn.Conv2d(ch_in,ch_in,kernel_size=1)

        self.norm=nn.Sequential(
            nn.InstanceNorm2d(ch_in),
        )
    def forward(self,x):
        b,c,h,w=x.size()
        x_norm=self.norm(x)
        x_split=self.to_qkv(x_norm)
        q,k,v=torch.split(x_split, self.ch_in , dim=1)
        # 将卷积后的张量沿着通道维度（dim=1）分割成三个部分，每个部分的通道数都是 ch_in。
        # 这三个部分分别对应查询（Q）、键（K）和值（V）。
        q = q.permute(0, 2, 3, 1).view(b, h * w, c) # [b,c,h,w]==>[b,h,w,c]==>[b,h*w,c] 以便后续的矩阵乘法操作
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        dot_products = torch.bmm(q, k) * (c ** (-0.5)) #* (c ** (-0.5))是为了对点积结果进行缩放,  [b,h*w,h*w]
        # 这种缩放是为了避免在计算注意力权重时，点积值过大导致 softmax 函数的梯度消失问题
        # torch.bmm   返回形状为 ((b, n, p)) 的张量，其中每个批次的矩阵是 input 和 mat2 对应批次矩阵的乘积
        # input 形状为 ((b, n, m)) ,mat2 形状为 ((b, m, p))。
        attention=torch.softmax(dot_products,dim=-1)  #在h*w维度上运用softmax  [b,h*w,h*w]
        out=torch.bmm(attention,v) #[h*w,h*w]*[h*w,c]==>[b,h*w,c]
        out=out.view(b,h,w,c).permute(0,3,1,2)
        return self.to_out(out) + x

test=torch.randn(12,3,32,32)
model=AttentionBlock(ch_in=3)
print(model(test).size())

class down_sample(nn.Module):  #下采样
    def __init__(self,ch_in):
        super(down_sample,self).__init__()

        self.down_sample=nn.Sequential(
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU(),
        )

    def forward(self,x):
        x=self.down_sample(x)
        return x

test=torch.randn(12,3,32,32)
model=down_sample(ch_in=3)
print(model(test).size())

class up_sample(nn.Module): #上采样
    def __init__(self,ch_in):
        super(up_sample,self).__init__()

        self.up_sample=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1),
        )

    def forward(self,x):
        x=self.up_sample(x)
        return x

test=torch.randn(12,3,32,32)
model_down=down_sample(ch_in=3)
down=model_down(test)
size=down.size(1)
model_up=up_sample(ch_in=size)
print(model_up(down).size())

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,time_emb_dim=None,num_classes=None,stride=1):
        super(ResBlk,self).__init__()

        self.bn=nn.BatchNorm2d(ch_in)

        self.conv=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=0),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=0),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU()
        )

        self.time_bias=nn.Linear(time_emb_dim,ch_out)
        self.classes_bias=nn.Embedding(num_classes,ch_out)if num_classes is not None else None
        self.attention=AttentionBlock(ch_out)

        self.shortcut=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=0),
            nn.InstanceNorm2d(ch_out)
        )

    def forward(self, x, time_emb=None, y=None):
        x = self.bn(x)
        out = self.conv(x)
        if time_emb is not None:
            out += self.time_bias(time_emb)[:, :, None, None] #[b,ch]==>[b,ch,1,1]
        if y is not None:
            out += self.classes_bias(y)[:, :, None, None]
        out = self.attention(out)
        output = out + self.shortcut(x)
        return output

test=torch.randn(12,3,32,32)
model=ResBlk(ch_in=3,ch_out=6,time_emb_dim=10,num_classes=10)
print(model(test).size())

class Unet(nn.Module):
    def __init__(self,dim,scale,time_dim,ch_in):
        super(Unet,self).__init__()

        #时间嵌入,将时间步长嵌入到模型中
        self.time_mlp=nn.Sequential(
            PositionEmbedding(dim=dim,scale=scale),
            nn.Linear(dim,time_dim),
            nn.SiLU(),  #SiLU，可以引入非线性变换，使得时间嵌入向量能够更好地表示复杂的时间模式
            nn.Linear(time_dim,time_dim)
        )

        self.conv1=nn.Conv2d(ch_in,dim,kernel_size=3,stride=1,padding=1) #初始层

        down=[]
        #下采样  每一次下采样加入两个残差块
        for i in range(4):
            for _ in range(2):
                down.append(ResBlk(ch_in=dim,ch_out=dim,time_emb_dim=time_dim))
            down.append(down_sample(ch_in=dim))

        self.down_sample=nn.Sequential(*down)

        self.mid_layer=nn.Sequential(
            ResBlk(ch_in=dim,ch_out=dim,time_emb_dim=time_dim),
            ResBlk(ch_in=dim,ch_out=dim,time_emb_dim=time_dim)

        )

        #上采样 每一次上采样加入两个残差块
        up=[]
        for i in range(4):
            for _ in range(2):
                up.append(ResBlk(ch_in=dim,ch_out=dim,time_emb_dim=time_dim))
            up.append(up_sample(ch_in=dim))

        self.up_sample=nn.Sequential(*up)
        self.change_ch=nn.Conv2d(dim*2,dim,kernel_size=3,stride=1,padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, ch_in, 3, padding=1,stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x,t):
        time_emb=self.time_mlp(t)
        x=self.conv1(x)
        # 下采样

        down_outputs = []
        for layer in self.down_sample:
            if isinstance(layer, ResBlk):
                x = layer(x, time_emb=time_emb)
            else:
                x = layer(x)
            down_outputs.append(x)
            # 中间层
        for layer in self.mid_layer:
            if isinstance(layer,ResBlk):
                x = layer(x, time_emb=time_emb)

            # 上采样
        for layer in self.up_sample:
            if isinstance(layer, ResBlk):
                # 先从 down_outputs 中取出一个张量，并进行上采样
                skip_connection = down_outputs.pop()  #pop取出down_outputs中最小的值
                if skip_connection.size() != x.size():
                    skip_connection = torch.nn.functional.interpolate(skip_connection, size=x.size()[2:])
                #torch.nn.functional.interpolate 函数的作用是对输入张量进行上采样或下采样。
                #在这里使用 size=x.size()[2:] 参数来指定目标大小，这样可以确保 skip_connection 的空间维度与 x 的空间维度一致。
                x = torch.cat((x, skip_connection), dim=1)  # [b,ch,x,x]==>[b,2ch,x,x]
                x=self.change_ch(x)
                #print('s',x.size())
                # print(x.size())
                x = layer(x, time_emb=time_emb)
            else:
                x = layer(x)
        
        x=self.out_conv(x)
        return x

model = Unet(dim=64, scale=1.0, time_dim=64, ch_in=3)
test=torch.randn(1,3,128,128)
t = torch.randn(1)  # 随机时间嵌入
print('ss',model(test,t).size())

def time_extract(t,v):
    device=t.device
    v=v.to(device)
    t=t.long()
    #将t扩展到与v的第一维度相同
    #t_expand=t.unsqueeze(1).expand(-1,v.size(1))
    out=torch.gather(v,index=t,dim=0).to(device)
    # torch.gather 会根据 t_expanded 中的索引(其中的值为索引)，从 v 的第一个维度上提取相应的值
    # v = torch.tensor([[0.1, 0.2, 0.3],
    #                   [0.4, 0.5, 0.6],
    #                   [0.7, 0.8, 0.9],
    #                   [1.0, 1.1, 1.2],
    #                   [1.3, 1.4, 1.5],
    #                   [1.6, 1.7, 1.8],
    #                   [1.9, 2.0, 2.1],
    #                   [2.2, 2.3, 2.4],
    #                   [2.5, 2.6, 2.7],
    #                   [2.8, 2.9, 3.0]])
    #
    # t_expanded = torch.tensor([[1, 1, 1],
    #                            [3, 3, 3],
    #                            [5, 5, 5]])
    #执行torch.gather(v, index=t_expanded, dim=0)后，结果将是：
    # tensor([[0.4, 0.5, 0.6],
    #         [1.0, 1.1, 1.2],
    #         [1.6, 1.7, 1.8]])
    #print(out.size())
    out=out.view(t.size(0),1,1,1)
    return out

v=torch.randn(12)
t=torch.arange(0,3)
print(time_extract(t,v).size())

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,model,beta_1,beta_T,T):
        super(GaussianDiffusionTrainer,self).__init__()

        self.model=model
        self.T=T

        self.betas=torch.linspace(beta_1,beta_T,T).float()
        #betas 是每个时间步长的噪声强度参数。通过 torch.linspace 生成的这些值，可以确保噪声强度在整个扩散过程中的变化是线性的。

        self.alphas=1.0-self.betas

        self.alphas_bar=torch.cumprod(self.alphas,dim=0) #alphas_bar 是 alphas 的累积乘积，用于表示从初始状态到当前时间步长的整体缩放因子

        self.sqrt_alphas_bar=torch.sqrt(self.alphas_bar)
        # sqrt_alphas_bar 是 alphas_bar 的平方根，用于在前向传播过程中缩放输入数据。

        self.sqrt_one_minus_alphas_bar=torch.sqrt(1.0-self.alphas_bar)

    def forward(self,x):
        t=torch.randint(self.T,(x.size(0),),device=x.device)
        noise=torch.randn_like(x)
        #print(self.sqrt_alphas_bar.size())
        x_t=(
            time_extract(t=t,v=self.sqrt_alphas_bar)*x+
            time_extract(t=t,v=self.sqrt_one_minus_alphas_bar)*noise
        )
        #将输入数据 ( x ) 和噪声 ( noise ) 结合起来，以模拟在时间步长 ( t ) 处的扩散状态。
        #print(x_t.size())
        loss=nn.functional.mse_loss(self.model(x_t,t),noise)
        #计算模型预测值(插入时间步长的图片)与实际噪声之间的均方误差
        #通过最小化这个损失，模型学习如何在不同时间步长下生成与实际噪声尽可能接近的预测值，从而在扩散过程中逐步去噪。
        return loss

model = Unet(dim=32,scale=1.0,time_dim=64,ch_in=3)
trainer = GaussianDiffusionTrainer(model, beta_1=0.1, beta_T=0.2, T=1000)
x_0 = torch.randn(16, 3, 64, 64)  # 示例输入
loss = trainer(x_0)
print(loss)


class GaussianDiffusionSampler(nn.Module):
    def __init__(self,beta_1,beta_T,T,model):
        super(GaussianDiffusionSampler,self).__init__()

        self.model_=model
        self.T=T

        self.betas=torch.linspace(beta_1,beta_T,T)

        self.alphas = 1.0 - self.betas

        self.alphas_bar = torch.cumprod(self.alphas, dim=0)  # alphas_bar 是 alphas 的累积乘积，用于表示从初始状态到当前时间步长的整体缩放因子

        self.alphas_bar_prev = nn.functional.pad(self.alphas_bar, [1, 0], value=1)[:T]
        #F.pad(alphas_bar, [1, 0], value=1)：在 alphas_bar 的前面填充一个值为 1 的元素。
        #F.pad 函数的 [1, 0] 参数表示在第一维度的前面填充一个元素，在后面不填充任何元素。value=1 表示填充值为 1。
        #这确保了在时间步长为 0 时，累积乘积为 1。
        #[1,0]:
        # 1 表示在张量的最后一个维度的前面填充一个元素。
        # 0 表示在张量的最后一个维度的后面不填充任何元素。
        # 在时间步长为 0 时，累积乘积 alphas_bar 应该为 1，因为这是初始状态，没有任何缩放。

        self.coeff1=torch.sqrt(1.0/self.alphas)
        #self.coeff1 是一个系数，用于在预测前一时间步长的均值时缩放当前时间步长的图像 x_t

        self.coeff2=self.coeff1*(1.0-self.alphas)/torch.sqrt(1.0-self.alphas_bar)
        #self.coeff2 是另一个系数，用于在预测前一时间步长的均值时缩放噪声 eps。
        #它结合了 coeff1、alphas 和 alphas_bar，确保在不同时间步长下正确地调整噪声的影响。

        self.posterior_var=self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        #self.posterior_var 表示后验方差，用于在采样过程中确定每个时间步长的噪声强度

    #根据当前时间步长的图像 x_t 和噪声 eps 预测前一时间步长的均值
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        return (
            time_extract(t=t,v=self.coeff1)*x_t - time_extract(t=t,v=self.coeff2)*eps
            #使用 coeff1 缩放 x_t。
            #使用 coeff2 缩放 eps。
        )

    def p_mean_variance(self, x_t, t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        #self.posterior_var[1:2]:
        #在时间步长为 1 时，后验方差的计算需要特别处理，因为没有前一时间步长的累积乘积。因此，选择 self.posterior_var 的第一个元素来处理这个特殊情况
        #取[1:2]是因为在self.alphas_bar最后一个维度前面添加了一个1
        #选择 self.betas[1:] 是为了确保在时间步长大于 1 时，能够正确地引用每个时间步长的噪声强度
        var = time_extract(t=t,v=var)

        eps = self.model_(x_t, t)

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self,x_T):
        x_t=x_T
        for time_step in reversed(range(self.T)):
            t=x_t.new_ones(x_T.size(0),dtype=torch.long)*time_step
            mean,var=self.p_mean_variance(x_t=x_t,t=t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
        x_0=x_t
        return torch.clip(x_0,-1,1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(dim=16,scale=1.0,time_dim=32,ch_in=3).to(device)
Sampler = GaussianDiffusionSampler(model=model, beta_1=0.1, beta_T=0.2, T=100).to(device)
x_0 = torch.randn(2, 3, 32, 32).to(device)  # 示例输入
x_0 = Sampler(x_0)
print(x_0.size())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(dim=16,scale=1.0,time_dim=32,ch_in=3).to(device)
tf=transforms.Compose([
    # transforms.Lambda(convert_to_rgb),
    transforms.Resize((32,32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data=datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=tf, download=True)
loader=DataLoader(data,batch_size=32,shuffle=True)

beta_1 = 1e-4
beta_T = 0.02
T = 1000
trainer = GaussianDiffusionTrainer(model=model, beta_1=beta_1, beta_T=beta_T, T=T)
sampler = GaussianDiffusionSampler(beta_1=beta_1, beta_T=beta_T, T=T, model=model)
optimizer = optim.Adam(trainer.parameters(), lr=1e-4)
for epoch in range(1000):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        batch_size=data.size(0)
        data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer.zero_grad()
        loss = trainer(data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f'Epoch [{epoch+1}/{1000}], Loss: {avg_loss:.4f}')

    # 每隔几个epoch进行一次采样
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            sample = sampler(torch.randn(batch_size, 3, 32, 32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            # 保存或展示采样结果
            # 保存到文件或使用matplotlib展示,这里省略
print("训练完成！")