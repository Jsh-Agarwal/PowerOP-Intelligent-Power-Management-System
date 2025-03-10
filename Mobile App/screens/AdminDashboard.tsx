import React, { useState, useEffect } from "react";
import { View, StyleSheet, ScrollView, TouchableOpacity, Modal, ActivityIndicator, Alert } from "react-native";
import { Text } from "react-native-elements";
import { useTheme } from "../components/ThemeContext";
import { DataTable } from "react-native-paper";
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = "http://localhost:8000";

const AdminDashboard = () => {
  const { theme } = useTheme();
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedAC, setSelectedAC] = useState<any | null>(null);
  const [acData, setAcData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchACData();
  }, []);

  const fetchACData = async () => {
    try {
      // Retrieve the token from AsyncStorage
      const token = await AsyncStorage.getItem('authToken');

      // Fetch Temperature Data
      const tempResponse = await fetch(`${API_BASE_URL}/api/temperature/current?device_id=test_device&zone_id=main`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const tempData = await tempResponse.json();

      // Fetch System Metrics
      const metricsResponse = await fetch(`${API_BASE_URL}/api/status/metrics?system_id=test_system`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const metricsData = await metricsResponse.json();

      if (tempData.temperature && metricsData.status === "success") {
        const updatedACData = [
          {
            id: 1,
            name: "Living Room AC",
            temp: tempData.temperature,
            energy: metricsData.data.summary.total_energy,
            power: metricsData.data.summary.avg_power,
            efficiency: metricsData.data.summary.efficiency,
            voltage: 220, // Hardcoded
            current: 4.5, // Hardcoded
            frequency: 50, // Hardcoded
          },
        ];
        setAcData(updatedACData);
      }
    } catch (error) {
      console.error("Error fetching AC data:", error);
      Alert.alert("Error", "Failed to fetch AC data.");
    } finally {
      setLoading(false);
    }
  };

  const openModal = (ac: any) => {
    setSelectedAC(ac);
    setModalVisible(true);
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme === "dark" ? "#121212" : "#FFFFFF",
      padding: 20,
    },
    title: {
      fontSize: 24,
      fontWeight: "bold",
      marginBottom: 20,
      color: theme === "dark" ? "#FFFFFF" : "#000000",
    },
    table: {
      backgroundColor: theme === "dark" ? "#1E1E1E" : "#F0F0F0",
    },
    tableHeader: {
      backgroundColor: theme === "dark" ? "#2C2C2C" : "#E0E0E0",
    },
    tableRow: {
      backgroundColor: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
    },
    text: {
      color: theme === "dark" ? "#FFFFFF" : "#000000",
    },
    button: {
      backgroundColor: theme === "dark" ? "#BB86FC" : "#6200EE",
      padding: 5,
      borderRadius: 5,
    },
    buttonText: {
      color: "#FFFFFF",
      textAlign: "center",
    },
    modalView: {
      margin: 20,
      backgroundColor: theme === "dark" ? "#1E1E1E" : "white",
      borderRadius: 20,
      padding: 35,
      alignItems: "center",
      shadowColor: "#000",
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.25,
      shadowRadius: 4,
      elevation: 5,
    },
    modalText: {
      marginBottom: 15,
      textAlign: "center",
      color: theme === "dark" ? "#FFFFFF" : "#000000",
    },
  });

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Admin Dashboard</Text>

      {loading ? (
        <ActivityIndicator size="large" color={theme === "dark" ? "#BB86FC" : "#6200EE"} />
      ) : (
        <DataTable style={styles.table}>
          <DataTable.Header style={styles.tableHeader}>
            <DataTable.Title>
              <Text style={styles.text}>AC Name</Text>
            </DataTable.Title>
            <DataTable.Title numeric>
              <Text style={styles.text}>Temp (°C)</Text>
            </DataTable.Title>
            <DataTable.Title>
              <Text style={styles.text}>Actions</Text>
            </DataTable.Title>
          </DataTable.Header>

          {acData.map((ac) => (
            <DataTable.Row key={ac.id} style={styles.tableRow}>
              <DataTable.Cell>
                <Text style={styles.text}>{ac.name}</Text>
              </DataTable.Cell>
              <DataTable.Cell numeric>
                <Text style={styles.text}>{ac.temp}</Text>
              </DataTable.Cell>
              <DataTable.Cell>
                <TouchableOpacity style={styles.button} onPress={() => openModal(ac)}>
                  <Text style={styles.buttonText}>View More</Text>
                </TouchableOpacity>
              </DataTable.Cell>
            </DataTable.Row>
          ))}
        </DataTable>
      )}

      {/* Modal for Detailed View */}
      <Modal animationType="slide" transparent={true} visible={modalVisible} onRequestClose={() => setModalVisible(false)}>
        <View style={styles.modalView}>
          {selectedAC && (
            <>
              <Text style={styles.modalText}>AC: {selectedAC.name}</Text>
              <Text style={styles.modalText}>Temperature: {selectedAC.temp}°C</Text>
              <Text style={styles.modalText}>Energy Consumed: {selectedAC.energy} kWh</Text>
              <Text style={styles.modalText}>Power: {selectedAC.power} W</Text>
              <Text style={styles.modalText}>Efficiency: {selectedAC.efficiency * 100}%</Text>
              <Text style={styles.modalText}>Voltage: {selectedAC.voltage} V</Text>
              <Text style={styles.modalText}>Current: {selectedAC.current} A</Text>
              <Text style={styles.modalText}>Frequency: {selectedAC.frequency} Hz</Text>
            </>
          )}
          <TouchableOpacity style={styles.button} onPress={() => setModalVisible(false)}>
            <Text style={styles.buttonText}>Close</Text>
          </TouchableOpacity>
        </View>
      </Modal>
    </ScrollView>
  );
};

export default AdminDashboard;