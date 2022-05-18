## Processos de instalação local (Linux) - LabIC 

1. **Instalação do Anaconda**

    Os comandos para o download podem ser encontrados no site oficial do Anaconda (<https://www.anaconda.com/products/distribution>).

2. **Criando um environment**

    No terminal: 

        conda create --name nome_do_environment

    Ativando o environment:
    
        conda activate nome_do_environment

3. **Verificando quais versões do CUDA e cuDNN o TensorFlow exige**

    Sendo a versão do TensorFlow  >= 2.5.0, o CUDA precisa ter a versão igual ou superior à 11.2.0. (<https://www.tensorflow.org/install/gpu?hl=pt-br#:~:text=Requisitos%20de%20software,-Os%20seguintes%20softwares&text=CUDA%C2%AE%20Toolkit%3A%20o%20TensorFlow,%3E%3D%202.5.0).>)

4. **Instalação do CUDA**

    Pesquisando as versões disponíveis do CUDA através do comando:

        conda search cudatoolkit

    Para essa pesquisa, foi baixada a versão 11.3.1, através do comando:

        conda install cudatoolkit=11.3.1

5. **Instalação do cuDNN**

    É importante verificar a compatibilidade entre as versões do cuDNN e do CUDA. Pesquisando as versões disponíveis do cuDNN através do comando:

        conda search cudnn

    Para essa pesquisa, foi baixada a versão 8.2.1, através do comando:

        conda install cudnn=8.2.1

6. **Instalação do TensorFlow**

    Pesquisando as versões disponíveis do TensorFlow:

        conda search tensorflow

    Para essa pesquisa, foi baixada a versão 2.6.0:

        conda install tensorflow=2.6.0

7. **Abrindo o environment pelo VS Code**

    Atalho:

    > CTRL + Shift + P

    Procure:

        Python: Select Interpreter

    Selecione o environment criado.




    

