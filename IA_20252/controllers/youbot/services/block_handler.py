class BlockHandler:
    """Classe para gerenciar operações de pegar e armazenar blocos"""
    
    def __init__(self, robo):
        """
        Inicializa o manipulador de blocos
        
        Args:
            robo: Instância do robô com arm e gripper
        """
        self.robo = robo
        self.counter = 0
        self.blocks_by_color = {
            "red": [],
            "green": [],
            "blue": []
        }
    
    def pick(self, label):
        """
        Pega o bloco da frente e coloca na base traseira do robô
        
        Args:
            label: Cor do bloco classificado ("red", "green", "blue")
        """
        
        print(f"Iniciando pick para bloco {label}...")
        
        # Incrementa o contador
        self.counter += 1
        
        if self.robo.arm.current_orientation != self.robo.arm.FRONT:
            self.robo.arm.set_orientation(self.robo.arm.FRONT)
            self.robo.wait(550)
        
        if self.robo.arm.current_height != self.robo.arm.RESET2:
            self.robo.arm.set_height(self.robo.arm.RESET2)
            self.robo.wait(550)
        
        # Abrir garra
        self.robo.gripper.release()
        self.robo.arm.set_height(self.robo.arm.FRONT_FLOOR)
        self.robo.wait(550)
        
        # Pegar o bloco
        self.robo.gripper.grip()
        self.robo.wait(300)
        print("Bloco agarrado!")
        
        # Levantar
        self.robo.arm.set_orientation(self.robo.arm.FRONT)
        self.robo.wait(550)
        
        # Vai para a pose de armazenamento
        self.robo.arm.set_pose(self.counter)
        self.robo.wait(550)
        
        # Soltar o bloco
        self.robo.gripper.release()
        self.robo.wait(300)
        
        # Registra o bloco na cor correspondente
        self.blocks_by_color[label].append(self.counter)
        print(f"Bloco {label} armazenado na posição {self.counter}!")
        
        self.robo.arm.reset(self.robo.arm.RESET2)
        self.robo.wait(200)
    
    def store(self, label):
        """
        Pega blocos de uma cor específica da base traseira e coloca na caixa
        
        Args:
            label: Cor dos blocos a serem armazenados ("red", "green", "blue")
        """
        
        blocos = self.blocks_by_color.get(label, [])
        
        if not blocos:
            print(f"Nenhum bloco {label} para armazenar!")
            return
        
        print(f"Armazenando {len(blocos)} bloco(s) {label}...")
        
        for pose_id in blocos:
            # Garantir posição segura
            if self.robo.arm.current_height != self.robo.arm.RESET2:
                self.robo.arm.set_height(self.robo.arm.RESET2)
                self.robo.wait(550)
            
            if self.robo.arm.current_orientation != self.robo.arm.FRONT:
                self.robo.arm.set_orientation(self.robo.arm.FRONT)
                self.robo.wait(550)
            
            # Abre a garra e desce pra pegar o bloco
            self.robo.gripper.release()
            self.robo.arm.set_pose(pose_id)
            self.robo.wait(600)

            # Agarrar bloco
            self.robo.gripper.grip()
            self.robo.wait(300)

            # Subir
            self.robo.arm.reset(self.robo.arm.RESET2)
            self.robo.wait(550)

            # Ajustar altura para soltar na caixa
            self.robo.arm.set_height(self.robo.arm.FRONT_CARDBOARD_BOX)
            self.robo.wait(600)

            # Soltar bloco
            self.robo.gripper.release()
            self.robo.wait(300)

            # Voltar para posição segura
            self.robo.arm.reset(self.robo.armRESET)
            self.robo.wait(550)
        
        # Limpa a lista da cor específica após armazenar
        self.blocks_by_color[label].clear()
        print(f"Blocos {label} armazenados na caixa!")
