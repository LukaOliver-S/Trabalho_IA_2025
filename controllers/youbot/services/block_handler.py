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
        self.stored = 0
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
        
        if self.robo.arm.current_orientation != self.robo.arm.FRONT:
            self.robo.arm.set_orientation(self.robo.arm.FRONT)
            self.robo.wait(700)
        
        if self.robo.arm.current_height != self.robo.arm.RESET2:
            self.robo.arm.set_height(self.robo.arm.RESET2)
            self.robo.wait(700)
        
        # Abrir garra
        self.robo.gripper.release()
        self.robo.arm.set_height(self.robo.arm.FRONT_FLOOR)
        self.robo.wait(600)
        
        # Pegar o bloco
        self.robo.gripper.grip()
        self.robo.wait(400)
        # current gap approx:0.07100000000000001
        print(self.robo.gripper.check_cube_grasped())
        if self.robo.gripper.has_object():
            print("Bloco agarrado!")
            self.counter+=1
            
            self.robo.arm.reset(self.robo.arm.RESET)
            self.robo.wait(400)
            
            # Vai para a pose de armazenamento
            self.robo.arm.set_pose(self.counter, "pick")
            self.robo.wait(900)
            
            # Soltar o bloco
            self.robo.gripper.release()
            self.robo.wait(300)
            
            # Registra o bloco na cor correspondente
            self.blocks_by_color[label].append(self.counter)
            print(f"Bloco {label} armazenado na posição {self.counter}!")
        else:
            print("Bloco não pego!")
        self.robo.arm.reset(self.robo.arm.RESET2)
        self.robo.wait(400)

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
                self.robo.wait(700)
            
            if self.robo.arm.current_orientation != self.robo.arm.FRONT:
                self.robo.arm.set_orientation(self.robo.arm.FRONT)
                self.robo.wait(700)
            
            # Abre a garra e desce pra pegar o bloco
            self.robo.gripper.release()
            self.robo.arm.set_pose(pose_id, "store")
            self.robo.wait(1000)

            # Agarrar bloco
            self.robo.gripper.grip()
            self.robo.wait(400)
            print(self.robo.gripper.check_cube_grasped())
            if self.robo.gripper.has_object():
                print("Bloco stored")
                self.stored+=1

            # Subir
            self.robo.arm.reset(self.robo.arm.RESET2)
            self.robo.wait(400)

            # Ajustar altura para soltar na caixa
            self.robo.arm.set_height(self.robo.arm.FRONT_CARDBOARD_BOX)
            self.robo.wait(900)

            # Soltar bloco
            self.robo.gripper.release()
            self.robo.wait(400)

            # Voltar para posição segura
            self.robo.arm.reset(self.robo.arm.RESET)
            self.robo.wait(400)
        
        # Limpa a lista da cor específica após armazenar
        self.blocks_by_color[label].clear()
        print(f"Blocos {label} armazenados na caixa!")
        
    def push_all_cubes(self):
        """
        Empurra todos os cubos usando sweep:
        - vai devagar para LEFT
        - varre rápido para RIGHT
        """
        print("🧹 Push all cubes com sweep poses")

        arm = self.robo.arm

        if self.robo.arm.current_height != self.robo.arm.RESET2:
            self.robo.arm.set_height(self.robo.arm.RESET2)
            self.robo.wait(700)

        # === IR DEVAGAR PARA LEFT ===
        arm.set_pose("0", mode="sweep", velocity=0.3)
        self.robo.wait(1000)

        # === VARREDURA RÁPIDA PARA RIGHT ===
        arm.set_pose("1", mode="sweep", velocity=1.57)
        self.robo.wait(700)

        # Voltar para posição segura
        arm.reset(arm.RESET2)
        self.robo.wait(500)

        # Resetar contadores
        self.counter = 0
        for color in self.blocks_by_color:
            self.blocks_by_color[color].clear()

        print("✅ Cubos empurrados com sucesso!")
