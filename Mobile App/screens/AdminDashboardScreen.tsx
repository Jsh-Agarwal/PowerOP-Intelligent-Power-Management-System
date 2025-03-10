"use client";

import { useState, useEffect } from "react";
import { View, ScrollView, TextStyle, ViewStyle, Alert } from "react-native";
import { Button, Text, ListItem, Switch } from "react-native-elements";
import { useTheme } from "../components/ThemeContext";
import { Ionicons } from "@expo/vector-icons";
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Define AC unit type with an additional power state
type ACUnit = {
  id: number;
  name: string;
  temp: number;
  energy: number;
  voltage: number;
  current: number;
  frequency: number;
  isOn: boolean; // Power state (on/off)
};

const AdminDashboardScreen = () => {
  const themeContext = useTheme();
  const theme = themeContext?.theme ?? "light";

  const [acUnits, setAcUnits] = useState<ACUnit[]>([
    { id: 1, name: "AC 1", temp: 22, energy: 1.5, voltage: 220, current: 4.5, frequency: 50, isOn: true },
    { id: 2, name: "AC 2", temp: 24, energy: 1.2, voltage: 220, current: 3.6, frequency: 50, isOn: true },
    { id: 3, name: "AC 3", temp: 23, energy: 1.8, voltage: 220, current: 5.4, frequency: 50, isOn: false },
  ]);

  const deviceId = "test_device"; // Replace with your actual device ID
  const zoneId = "main"; // Replace with your actual zone ID

  useEffect(() => {
    fetchCurrentTemperature();
  }, []);

  const fetchCurrentTemperature = async () => {
    try {
      // Retrieve the token from AsyncStorage
      const token = await AsyncStorage.getItem('authToken');

      const response = await axios.get(`http://localhost:8000/api/temperature/current`, {
        headers: {
          Authorization: `Bearer ${token}`, // Use the retrieved token
        },
        params: {
          device_id: deviceId,
          zone_id: zoneId,
        },
      });

      // Update the AC units with the current temperature
      setAcUnits((prevUnits) =>
        prevUnits.map((unit) => ({
          ...unit,
          temp: response.data.temperature, // Set the current temperature
        }))
      );
    } catch (error) {
      console.error('Error fetching current temperature:', error);
      Alert.alert('Error', 'Failed to fetch current temperature.');
    }
  };

  const adjustTemperature = async (id: number, adjustment: number) => {
    setAcUnits((prevUnits) =>
      prevUnits.map((unit) => {
        if (unit.id === id) {
          const newTemp = unit.temp + adjustment;
          // Call the API to set the new temperature
          setTemperature(unit.id, newTemp, "cool"); // Pass a string for the mode
          return { ...unit, temp: newTemp };
        }
        return unit;
      })
    );
  };

  const setTemperature = async (id: number, temperature: number, mode: string) => {
    try {
      // Retrieve the token from AsyncStorage
      const token = await AsyncStorage.getItem('authToken');

      const response = await axios.post(`http://localhost:8000/api/control/temperature`, {
        system_id: "test_system", // Replace with your actual system ID
        temperature: temperature,
        mode: mode, // Ensure this is a string
      }, {
        headers: {
          Authorization: `Bearer ${token}`, // Use the retrieved token
        },
      });
      console.log('Temperature set successfully:', response.data);
    } catch (error) {
      console.error('Error setting temperature:', error);
      Alert.alert('Error', 'Failed to set temperature.');
    }
  };

  const togglePower = async (id: number) => {
    setAcUnits((prevUnits) =>
      prevUnits.map((unit) => {
        if (unit.id === id) {
          const newState = !unit.isOn; // This is a boolean
          controlPower(unit.id, newState); // Ensure controlPower accepts a boolean
          return { ...unit, isOn: newState };
        }
        return unit;
      })
    );
  };

  const controlPower = async (id: number, state: boolean) => { // Ensure state is boolean
    try {
      // Retrieve the token from AsyncStorage
      const token = await AsyncStorage.getItem('authToken');

      const response = await axios.post(`http://localhost:8000/api/control/power`, {
        system_id: "test_system", // Replace with your actual system ID
        state: state, // This should be a boolean
      }, {
        headers: {
          Authorization: `Bearer ${token}`, // Use the retrieved token
        },
      });
      console.log('Power state changed successfully:', response.data);
    } catch (error) {
      console.error('Error changing power state:', error);
      Alert.alert('Error', 'Failed to change power state.');
    }
  };

  return (
    <ScrollView style={styles.container(theme)}>
      <Text style={styles.title(theme)}>Admin Control</Text>

      {/* AC Units Section */}
      <View style={styles.section(theme)}>
        <Text style={styles.subtitle(theme)}>AC Units</Text>
        {acUnits.map((ac) => (
          <ListItem key={ac.id} bottomDivider containerStyle={styles.listItem(theme)}>
            <ListItem.Content>
              <ListItem.Title style={styles.text(theme)}>{ac.name}</ListItem.Title>
              <View style={styles.row}>
                <Text style={styles.text(theme)}>{ac.temp}°C</Text>
                <View style={styles.row}>
                  <Button
                    icon={<Ionicons name="remove" size={24} color={theme === "dark" ? "#fff" : "#000"} />}
                    type="clear"
                    onPress={() => adjustTemperature(ac.id, -1)}
                  />
                  <Button
                    icon={<Ionicons name="add" size={24} color={theme === "dark" ? "#fff" : "#000"} />}
                    type="clear"
                    onPress={() => adjustTemperature(ac.id, 1)}
                  />
                </View>
              </View>

              {/* Space between temperature control and on/off toggle */}
              <View style={{ marginVertical: 10 }} />

              {/* On/Off Toggle */}
              <View style={styles.row}>
                <Text style={styles.text(theme)}>{ac.isOn ? "On" : "Off"}</Text>
                <Switch
                  value={ac.isOn}
                  onValueChange={() => togglePower(ac.id)}
                  color={theme === "dark" ? "#fff" : "#000"}
                />
              </View>
            </ListItem.Content>
          </ListItem>
        ))}
      </View>

      {/* System Overview Section */}
      <View style={styles.section(theme)}>
        <Text style={styles.subtitle(theme)}>System Overview</Text>
        <Text style={styles.text(theme)}>
          Total Energy Consumption: {acUnits.reduce((sum, ac) => sum + ac.energy, 0).toFixed(2)} kWh
        </Text>
        <Text style={styles.text(theme)}>
          Average Temperature: {(acUnits.reduce((sum, ac) => sum + ac.temp, 0) / acUnits.length).toFixed(1)}°C
        </Text>
      </View>
    </ScrollView>
  );
};

// Styles with dynamic theming and additional design improvements
const styles = {
  container: (theme: string): ViewStyle => ({
    flex: 1,
    backgroundColor: theme === "dark" ? "#121212" : "#FFFFFF",
    padding: 20,
  }),
  title: (theme: string): TextStyle => ({
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
    color: theme === "dark" ? "#FFFFFF" : "#000000",
  }),
  section: (theme: string): ViewStyle => ({
    marginBottom: 20,
    backgroundColor: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
    padding: 15,
    borderRadius: 10,
    shadowColor: theme === "dark" ? "#fff" : "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 5,
  }),
  subtitle: (theme: string): TextStyle => ({
    fontSize: 18,
    fontWeight: "600",
    color: theme === "dark" ? "#FFFFFF" : "#000000",
    marginBottom: 10,
  }),
  listItem: (theme: string): ViewStyle => ({
    backgroundColor: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
    borderRadius: 8,
  }),
  text: (theme: string): TextStyle => ({
    color: theme === "dark" ? "#FFFFFF" : "#000000",
  }),
  row: {
    flexDirection: "row" as ViewStyle["flexDirection"],
    justifyContent: "space-between" as ViewStyle["justifyContent"],
    alignItems: "center" as ViewStyle["alignItems"],
  } as ViewStyle,
};

export default AdminDashboardScreen;