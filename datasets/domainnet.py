import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class DOMAINNET(DatasetBase):
    """DOMAINNET
    """

    dataset_dir = "DOMAINNET"
    domains = ["clipart", "infograph", "painting", "sketch","real"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)


    def _read_data(self, input_domains):
        items = []

        # 创建全局类别到 label 的映射
        global_class_names = set()
        for dname in input_domains:
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            global_class_names.update(class_names)  # 合并每个域的类别到全局类别集合

        # 排序并为每个类别分配唯一的 label
        global_class_names = sorted(global_class_names)
        self.global_lab2cname = {label: cname for label, cname in enumerate(global_class_names)}
        print('global_lab2cname:', self.global_lab2cname)

        # 对每个域的数据进行处理
        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()
            print(f'class_names in {dname}:', class_names)

            for class_name in class_names:
                label = global_class_names.index(class_name)  # 根据全局类别列表获取 label
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)

        return items



    #下文为原来的域lab2cname处理方式
    '''def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()
            print('class_names:', class_names)

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)
                    #print('item.label:', item.label)
        return items'''
