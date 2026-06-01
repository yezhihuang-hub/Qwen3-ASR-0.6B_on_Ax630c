# Pulsar2 发布说明

典型的发布目录结构如下

```bash
├── ax_pulsar2_${version}_diff_pulsar2_${diff_version}.tar.gz
├── ax_pulsar2_${version}_doc.tar.gz
├── ax_pulsar2_${version}_package.tar.gz
├── ax_pulsar2_${version}.tar.gz
└── README.md
```

接下来将分解说明每个文件的作用和使用方法，执行命令之前请将 `version` 替换为对应的 `Pulsar2` 版本。

### `ax_pulsar2_${version}.tar.gz`

`Pulsar2` 工具链 docker 镜像。

通过命令 `docker load -i ax_pulsar2_${version}.tar.gz` 加载镜像。

使用 docker 镜像进行模型转换及编译的方法详情，请参考 `Pulsar2 文档`。

### `ax_pulsar2_${version}_diff_pulsar2_${diff_version}.tar.gz`

`Pulsar2` 工具链 docker 镜像增量更新包。

当服务器本地已经加载过 `${diff_version}` 版本或者以上版本的 `Pulsar2` 镜像以后，可以通过命令 `docker load -i ax_pulsar2_${version}_diff_pulsar2_${diff_version}.tar.gz` 进行增量更新。

此增量更新包大小远小于完整的 docker 镜像，为 MB 级别，可以极大程度上提升升级效率。

### `ax_pulsar2_${version}_doc.tar.gz`

`Pulsar2` 工具链文档说明。

通过命令 `tar -zxvf ax_pulsar2_${version}_doc.tar.gz` 解压后，进入 `public` 目录，用浏览器打开 `index.html` 即可查看 `Pulsar2` 工具链文档。

### `ax_pulsar2_${version}_package.tar.gz`

`Pulsar2`工具链安装包，此安装包与 docker 镜像都可以用于模型转换及编译，但是形态不同。

安装包里内置了 `Pulsar2` 工具链运行所需的全部依赖，用户只需确保安装了正确 Python 版本就可以运行工具链。

通过命令 `tar -zxvf ax_pulsar2_${version}_package.tar.gz` 解压以后，查看解压目录中的 `README.md` 查看使用方法。
