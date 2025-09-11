import axios from "axios";

const API_BASE = "http://localhost:5173/";

export async function getTrafficStatus(intersectionId) {
  try {
    const resp = await axios.get(`${API_BASE}/traffic/status/${intersectionId}`);
    return resp.data;
  } catch (error) {
    console.error("API error:", error);
    return null;
  }
}

export async function getSignalStatus(intersectionId) {
  try {
    const resp = await axios.get(`${API_BASE}/signal/status/${intersectionId}`);
    return resp.data;
  } catch (error) {
    console.error("API error:", error);
    return null;
  }
}