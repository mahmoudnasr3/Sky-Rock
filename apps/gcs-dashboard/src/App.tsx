import { useEffect, useState } from "react";
import "./App.css";

type HealthResponse = {
  status: string;
};

type ClassesResponse = {
  classes: Record<string, string>;
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [classes, setClasses] = useState<ClassesResponse | null>(null);

  async function loadBackend() {
    const healthResponse = await fetch("http://127.0.0.1:8000/health");
    const healthData = await healthResponse.json();
    setHealth(healthData);

    const classesResponse = await fetch("http://127.0.0.1:8000/classes");
    const classesData = await classesResponse.json();
    setClasses(classesData);
  }

  useEffect(() => {
    loadBackend().catch(console.error);
  }, []);

  return (
    <main className="container">
      <section className="card">
        <h1>Sky-Rock GCS Dashboard</h1>
        <p>Safe aerial robotics monitoring dashboard.</p>
      </section>

      <section className="card">
        <h2>Backend Status</h2>
        <p>{health ? health.status : "Loading..."}</p>
      </section>

      <section className="card">
        <h2>Dataset Classes</h2>
        {classes ? (
          <ul>
            {Object.entries(classes.classes).map(([id, name]) => (
              <li key={id}>
                {id}: {name}
              </li>
            ))}
          </ul>
        ) : (
          <p>Loading classes...</p>
        )}
      </section>
    </main>
  );
}

export default App;