import json
from pathlib import Path

from web3 import Web3


BLOCKCHAIN_RPC = "http://127.0.0.1:8545"

PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
ACCOUNT_ADDRESS = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"

DEPLOYMENT_DIR = Path("blockchain/deployment")

CONTRACT_ADDRESS_PATH = DEPLOYMENT_DIR / "contract-address.json"
CONTRACT_ABI_PATH = DEPLOYMENT_DIR / "contract-abi.json"


class BlockchainService:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC))
        self.contract = None

    def is_available(self):
        return self.web3.is_connected()

    def load_contract(self):
        if self.contract is not None:
            return self.contract

        if not CONTRACT_ADDRESS_PATH.exists():
            raise FileNotFoundError(f"Missing {CONTRACT_ADDRESS_PATH}")

        if not CONTRACT_ABI_PATH.exists():
            raise FileNotFoundError(f"Missing {CONTRACT_ABI_PATH}")

        with open(CONTRACT_ADDRESS_PATH, "r", encoding="utf-8") as f:
            address = json.load(f)["address"]

        with open(CONTRACT_ABI_PATH, "r", encoding="utf-8") as f:
            abi = json.load(f)

        self.contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=abi,
        )

        return self.contract

    def log_detection(self, source_id, class_name, confidence, metadata_hash):
        if not self.is_available():
            raise ConnectionError("Blockchain node is not available")

        contract = self.load_contract()

        nonce = self.web3.eth.get_transaction_count(ACCOUNT_ADDRESS)

        tx = contract.functions.logDetection(
            source_id,
            class_name,
            int(confidence * 10000),
            metadata_hash,
        ).build_transaction(
            {
                "from": ACCOUNT_ADDRESS,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": self.web3.to_wei("2", "gwei"),
            }
        )

        signed_tx = self.web3.eth.account.sign_transaction(
            tx,
            private_key=PRIVATE_KEY,
        )

        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
        }