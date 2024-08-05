# 使用 Ansible 将 SSH 公钥分发到多个目标主机
## 0. 安装Ansible

```bash
pip install ansible-vault
```

## 1. 创建变量文件并加密

创建一个包含密码的变量文件vars.yml：

```yaml
all:
  hosts:
    192.168.1.27:
      ansible_user: myuser
      ansible_password: mypassword
    192.168.1.28:
      ansible_user: myuser
      ansible_password: mypassword
```

然后使用Ansible Vault加密这个文件：

```bash
ansible-vault encrypt vars.yml
```

注意：

1. 执行 `ansible-vault` 的过程中需要设定一个密码，请记住或保存好这个密码
2. `vars.yml`将被替换为加密后的文件

## 2. 创建主机清单文件

创建一个主机清单文件`inventory.ini`：

```ini
[all]
node1 ansible_host=192.168.1.27 ansible_user=myuser
node2 ansible_host=192.168.1.28 ansible_user=myuser
```

注：需要根据情况修改 `ansible_user` 的值

## 3. 创建Playbook

如果文件存在，这一步可以忽略。

创建一个Playbook distribute_ssh_key.yml：

```yaml
---
- name: Distribute SSH key
  hosts: all
  vars_files:
    - vars.yml
  tasks:
    - name: Create .ssh directory if it doesn't exist
      file:
        path: /home/{{ ansible_user }}/.ssh
        state: directory
        mode: '0700'
        owner: "{{ ansible_user }}"
        group: "{{ ansible_user }}"

    - name: Copy the SSH key to the authorized_keys file
      authorized_key:
        user: "{{ ansible_user }}"
        state: present
        key: "{{ lookup('file', '/path/to/id_rsa.pub') }}"
```

注：`vars_files` 配置为 `vars.yml`

## 4. 运行Playbook

使用以下命令运行Playbook，并解密变量文件：

```bash
ansible-playbook -i inventory.ini distribute_ssh_key.yml --ask-vault-pass
```
或者运行

```bash
./dist_ssh_key.sh
```

