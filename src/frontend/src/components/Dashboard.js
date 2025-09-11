import React, { useEffect, useState } from "react";
import { getTrafficStatus,  getSignalStatus } from "../utils/api";

function Dashboard({ intersectionId }) {
  const [trafficData, setTrafficData] = useState(null);
  const [signalData, setSignalData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      const traffic = await getTrafficStatus(intersectionId);
      setTrafficData(traffic?.data || null);

      const signal = await getSignalStatus(intersectionId);
      setSignalData(signal?.data || null);
    }
    fetchData();
  }, [intersectionId]);

  if (!trafficData || !signalData) return <p>Loading data...</p>;

  return (
    <div>
      <h2>Intersection: {intersectionId}</h2>
      <h3>Traffic Counts</h3>
      <pre>{JSON.stringify(trafficData.lane_counts, null, 2)}</pre>

      <h3>Signal Timings</h3>
      <pre>{JSON.stringify(signalData.current_timings || signalData.optimized_timings, null, 2)}</pre>
    </div>
  );
}

export default Dashboard;
