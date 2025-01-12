1. 数据路径问题

self.vox_files = glob.glob('../{0}/{1}/*/*.vox'.format(self.vox_path, self.vox_type))改成
self.vox_files = glob.glob(os.path.join(self.vox_path, self.vox_type, '*', '*.vox'))

2. float和double

vox = self.__read_vox__(img_path).astype(np.float32)

3. import tqdm要from tqdm import tqdm 不然报错

4. bceloss只能是0-1

5. G_forward out的view问题shape

6. nn.sigmoid要先实例化，才能用作计算函数

7. labels要和D的输出形状一样

8. 计算图gloss无法backward
是因为训练完D之后zero_grad了G，导致丢失计算图。解决：第二次再进行一次frag+G(frag)
