import { ethers } from "ethers";
import fs from "fs";
import path from "path";

async function main() {
  const artifactPath = path.join(
    process.cwd(),
    "artifacts",
    "contracts",
    "SkyRockLogger.sol",
    "SkyRockLogger.json"
  );

  const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));

  const provider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");

  const privateKey =
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

  const wallet = new ethers.Wallet(privateKey, provider);

  const factory = new ethers.ContractFactory(
    artifact.abi,
    artifact.bytecode,
    wallet
  );

  const contract = await factory.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();

  console.log("SkyRockLogger deployed to:", address);

  const deploymentDir = path.join(process.cwd(), "deployment");

  fs.mkdirSync(deploymentDir, { recursive: true });

  fs.writeFileSync(
    path.join(deploymentDir, "contract-address.json"),
    JSON.stringify({ address }, null, 2)
  );

  fs.writeFileSync(
    path.join(deploymentDir, "contract-abi.json"),
    JSON.stringify(artifact.abi, null, 2)
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});