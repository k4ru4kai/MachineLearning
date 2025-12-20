# --- 1. CONFIGURAZIONE AMBIENTE ---
import os

# DISABILITA LA GPU PER TENSORFLOW (Fondamentale su Ubuntu 24.04 con ROS)
# -1 nasconde tutte le schede video a TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- 2. ORA PUOI IMPORTARE TUTTO IL RESTO ---
import rclpy
from rclpy.node import Node
# ... (resto degli import) ...
import tensorflow as tf

# --- 2. IMPORT ROS 2 (Prima di TensorFlow!) ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tf2_ros
from tf2_ros import TransformException

# --- 3. LIBRERIE SCIENTIFICHE ---
import joblib
import numpy as np

# --- CONFIGURAZIONE ---
# Inserire nome della cartella 
EXPERIMENT_FOLDER = 'Reacher3_reacher3_train_1_512-512-256_dropYes_400ep' # <--- CAMBIA QUESTO con il nome della tua cartella!

# Il resto lo costruisce da solo
BASE_PATH = f'results/{EXPERIMENT_FOLDER}'
MODEL_PATH = f'{BASE_PATH}/model.h5'
SCALER_X_PATH = f'{BASE_PATH}/scaler_x.pkl' 
SCALER_Y_PATH = f'{BASE_PATH}/scaler_y.pkl'

class NeuralController(Node):
    def __init__(self):
        super().__init__('neural_controller')
        
        # 1. CARICAMENTO DEL CERVELLO
        self.get_logger().info('🧠 Caricamento Rete Neurale...')
        
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.scaler_x = joblib.load(SCALER_X_PATH)
            self.scaler_y = joblib.load(SCALER_Y_PATH)
            self.get_logger().info('✅ Modello caricato!')
        except Exception as e:
            self.get_logger().error(f'❌ Errore caricamento file: {e}')
            self.get_logger().error(f'Controlla di avere i file .h5 e .pkl corretti in {os.getcwd()}/results')
            sys.exit(1)

        # 2. ROS SETUP
        # Publisher verso i motori
        self.pub_cmd = self.create_publisher(Float64MultiArray, '/arm_position_controller/commands', 10)
        
        # Subscriber per leggere i giunti attuali
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # TF Listener per sapere dove si trova la "mano" (End Effector)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Stato interno
        self.q = np.zeros(3) # Angoli correnti (Reacher3 ha 3 giunti)
        self.x = None        # Posizione cartesiana corrente
        
        # Target Iniziale (Un punto a caso raggiungibile)
        self.target = np.array([0.5, 0.5]) 
        
        # Timer: Esegue il loop di controllo 10 volte al secondo (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('🚀 Controller avviato. In attesa di dati...')

    def joint_callback(self, msg):
        # Legge gli angoli q dal simulatore
        # Filtra solo i giunti che ci interessano (arm_joint_1, arm_joint_2...)
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if 'arm_joint' in name:
                positions.append(pos)
        
        # Assicuriamoci di averne letti 3
        if len(positions) >= 3:
            self.q = np.array(positions[:3])

    def get_end_effector_pos(self):
        # Chiede a TF la posizione della punta del robot
        try:
            # Cerca la trasformazione dalla base alla punta
            # [cite_start]'tip_link' è il nome standard nel pacchetto marr_gz [cite: 418]
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'tip_link',
                rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y])
        except TransformException:
            return None

    def control_loop(self):
        # 1. Leggi dove siamo (X)
        self.x = self.get_end_effector_pos()
        if self.x is None:
            return # Aspetta che TF sia pronto

        # 2. Calcola Distanza dal Target
        dt = self.target - self.x
        dist = np.linalg.norm(dt)
        
        # Se siamo arrivati (errore < 5cm), cambia target!
        if dist < 0.05:
            self.get_logger().info(f'🎉 Target Raggiunto! (Err: {dist:.3f})')
            # Nuovo target random (in un range raggiungibile)
            self.target = np.random.uniform(-0.6, 0.6, 2)
            self.get_logger().info(f'🎯 Nuovo Target: {self.target}')
            return

        # 3. Step Scaling (Normalizzazione del passo)
        # La rete è addestrata su piccoli spostamenti. Se il target è lontano, 
        # facciamo solo un passo in quella direzione.
        max_step = 0.05
        if dist > max_step:
            dx = dt / dist * max_step
        else:
            dx = dt

        # 4. PREPARAZIONE INPUT (Feature Engineering)
        # Importante: Deve essere UGUALE a come hai addestrato la rete!
        # Input: [x, y, sin(q1,q2,q3), cos(q1,q2,q3), dx, dy]
        
        sin_q = np.sin(self.q)
        cos_q = np.cos(self.q)
        
        # Concateniamo tutto in un vettore da 10 elementi
        input_vec = np.concatenate([self.x, sin_q, cos_q, dx])
        
        # 5. PREDIZIONE NEURALE
        # Scaliamo l'input (Robot -> Rete)
        input_scaled = self.scaler_x.transform([input_vec])
        
        # Chiediamo alla rete il dq
        dq_scaled = self.model.predict(input_scaled, verbose=0)
        
        # Scaliamo l'output (Rete -> Robot)
        dq = self.scaler_y.inverse_transform(dq_scaled)[0]

        # 6. INVIA COMANDO
        # Calcoliamo la nuova configurazione desiderata
        new_q = self.q + dq
        
        # Inviamo il comando a ROS
        msg = Float64MultiArray()
        msg.data = new_q.tolist()
        self.pub_cmd.publish(msg)
        
        # Debug su terminale (opzionale)
        # print(f"Dist: {dist:.3f} | Cmd: {dq}")

def main(args=None):
    rclpy.init(args=args)
    node = NeuralController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()