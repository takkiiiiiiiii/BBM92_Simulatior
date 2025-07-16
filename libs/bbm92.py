from qiskit import QuantumCircuit
from qiskit_aer.noise import (NoiseModel, pauli_error)

from qiskit_aer import AerSimulator
# from kr_Hamming import key_reconciliation_Hamming
from IPython.display import display
# from qiskit.tools.visualization import plot_histogram
import numpy as np


class Repeater:
   
    def __init__(self, name: str):
        self.name = name
        self.sharekeys = []

    def manage_sharekey(self, new_sharekey: str):
        self.sharekeys.append(new_sharekey)
        print(f"新しい共有鍵が追加されました。現在の鍵リスト: {self.sharekeys}")

class User:
    def __init__(self, name: str):
        self.name = name
        self.sharekeys = ''
    
    def __repr__(self):
        return f"User(name='{self.name}')"


def generate_Siftedkey(num_qubits, p_error_repeater, p_error_user, backend):
    bell_pair = int(num_qubits/2)
    repeater_basis = qrng(bell_pair, backend)
    user_basis = qrng(bell_pair, backend)

    # Compose the quantum circuit to generate the Bell state
    qc = compose_quantum_circuit(num_qubits)

    for i in range(num_qubits):
        qc.id(i)

    repeater_qubits = [i for i in range(0, num_qubits, 2)]
    user_qubits = [i for i in range(1, num_qubits, 2)]

    # Apply the quantum error chanel
    noise_model = apply_x_basis_measurement_noise(p_error_repeater, p_error_user, repeater_qubits, user_qubits)

    # 測定（IDゲートにノイズをかけて、Xエラーとして再現）
    qc, repeater_bits, user_bits = alice_bob_measurement(qc, num_qubits, noise_model, backend)

    ab_basis = check_bases(repeater_basis,user_basis)

    repeater_siftedkey = gen_alicekey(repeater_bits, ab_basis)
    user_siftedkey = gen_bobkey(user_bits, ab_basis)

    err_num = 0
    err_num = sum(1 for a, b in zip(repeater_siftedkey, user_siftedkey) if a != b)

    return repeater_siftedkey, user_siftedkey, err_num


def qrng(n, backend):
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)  # Apply Hadamard gate
    qc.measure(range(n), range(n))
    job_result = backend.run(qc, shots=1).result()
    counts = job_result.get_counts()

    # 取得した測定結果の中で最も出現回数が多いものを採用
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))  # ビット列を逆順にして取得

    return bits

# Generate bell state (Need 2 qubits per a state)
def get_bellState(n):
    qc = QuantumCircuit(n,n) 
    for i in range(0, n, 2):
        # i: corresponds to Alice's qubit.
        # i+1: corresponds to Bob's qubit.
        qc.x(i+1) # Pauli-X gate 
        qc.h(i) # Hadamard gate 
        qc.cx(i,i+1) # CNOT gate
    qc.barrier()
    return qc

def compose_quantum_circuit(num_qubit) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubit, num_qubit)
    qc.compose(get_bellState(num_qubit), inplace=True)
    return qc


def compose_quantum_circuit_for_eve(num_qubit) -> QuantumCircuit:
    qc2 = QuantumCircuit(num_qubit, num_qubit)
    qc2.compose(get_bellState(num_qubit), inplace=True)
    return qc2


def apply_x_basis_measurement_noise(
    p_error_repeater,
    p_error_user,
    x_basis_qubits_repeater,
    x_basis_qubits_user
):
    noise_model = NoiseModel()
    
    # エラーモデルを定義
    repeater_error = pauli_error([('X', p_error_repeater), ('I', 1 - p_error_repeater)])
    user_error = pauli_error([('X', p_error_user), ('I', 1 - p_error_user)])
    
    # リピーター側のX基底qubitの "measure" にエラーを適用
    for qubit in x_basis_qubits_repeater:
        noise_model.add_quantum_error(repeater_error, 'measure', [qubit])
        
    # ユーザー側のX基底qubitの "measure" にエラーを適用
    for qubit in x_basis_qubits_user:
        noise_model.add_quantum_error(user_error, 'measure', [qubit])
        
    return noise_model


def alice_bob_measurement(qc, num_qubits, noise_model, backend):
    qc.barrier()
    qc.measure(list(range(num_qubits)), list(range(num_qubits)))

    # IDゲートを挿入することで、特定量子ビットにnoise_modelが作用可能に（Xノイズのトリガー）
    for qubit in range(num_qubits):
        qc.id(qubit)

    result = backend.run(qc, shots=1, noise_model=noise_model).result()
    counts = result.get_counts()
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))
    alice_bits = bits[::2]
    bob_bits = bits[1::2]

    return [qc, alice_bits, bob_bits]


# check where bases matched
def check_bases(b1,b2):
    check = ''
    # matches = 0
    for i in range(len(b1)):
        if b1[i] == b2[i]: 
            check += "Y" 
            # matches += 1
        else:
            check += "-"
    return check

# check where measurement bits matched
def check_bits(b1,b2,bck):
    check = ''
    for i in range(len(b1)):
        if b1[i] == b2[i] and bck[i] == 'Y':
            check += 'Y'
        elif b1[i] == b2[i] and bck[i] != 'Y':
            check += 'R'
        elif b1[i] != b2[i] and bck[i] == 'Y':
            check += '!'
        elif b1[i] != b2[i] and bck[i] != 'Y':
            check += '-'

    return check

def gen_alicekey(bits, ab_bases):
    alice_sifted_key = ''
    for i in range(len(bits)):
        if ab_bases[i] == 'Y':
            alice_sifted_key += bits[i]
    return alice_sifted_key


def gen_bobkey(bits, ab_bases):
    bob_sifted_key = ''
    for i in range(len(bits)):
        if ab_bases[i] == 'Y':
            # bits[i] を反転
            flipped_bit = '1' if bits[i] == '0' else '0'
            bob_sifted_key += flipped_bit
    return bob_sifted_key



def main():
    backend = AerSimulator()
    num_qubits = 100
    p_error_repeater = 0.05
    p_error_user = 0.04
    repeater_siftedkey, user_siftedkey, err_num = generate_Siftedkey(num_qubits, p_error_repeater, p_error_user, backend)
    print(f"Repeater's (Alice's) Key:      {repeater_siftedkey}")
    print(f"User's Key:                    {user_siftedkey}")
    print(f'QQBER: {err_num/len(repeater_siftedkey)}')


if __name__ == '__main__':
    main()



