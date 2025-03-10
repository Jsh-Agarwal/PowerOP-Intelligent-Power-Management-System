import React, { useState, useEffect } from "react";
import { View, Text, ScrollView, StyleSheet, Switch, Animated } from "react-native";
import { Button } from "react-native-elements";
import { Picker } from "@react-native-picker/picker";
import { Ionicons } from "@expo/vector-icons";
import axios from 'axios';

const HomeScreen = () => {
  const [selectedAC, setSelectedAC] = useState("1");
  const [acUnits, setAcUnits] = useState([
    { id: "1", name: "AC 1", temp: 22, mode: "Cool", isOn: true },
    { id: "2", name: "AC 2", temp: 24, mode: "Heat", isOn: false },
    { id: "3", name: "AC 3", temp: 23, mode: "Fan", isOn: true },
  ]);

  const [weatherData, setWeatherData] = useState({
    temperature: '--',
    humidity: '--',
    windSpeed: '--',
    precipitation: '--',
  });

  const animatedValue = new Animated.Value(0);
  
  const animateChange = () => {
    Animated.sequence([
      Animated.timing(animatedValue, { toValue: 1, duration: 300, useNativeDriver: true }),
      Animated.timing(animatedValue, { toValue: 0, duration: 300, useNativeDriver: true }),
    ]).start();
  };

  const adjustTemperature = (adjustment: number) => {
    setAcUnits((prevUnits) =>
      prevUnits.map((unit) =>
        unit.id === selectedAC ? { ...unit, temp: unit.temp + adjustment } : unit
      )
    );
    animateChange();
  };

  const togglePower = () => {
    setAcUnits((prevUnits) =>
      prevUnits.map((unit) =>
        unit.id === selectedAC ? { ...unit, isOn: !unit.isOn } : unit
      )
    );
  };

  const changeMode = (newMode: string) => {
    setAcUnits((prevUnits) =>
      prevUnits.map((unit) =>
        unit.id === selectedAC ? { ...unit, mode: newMode } : unit
      )
    );
  };

  const fetchWeatherData = async () => {
    const latitude = '28.7041'; // Example latitude for New Delhi
    const longitude = '77.1025'; // Example longitude for New Delhi
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation`;

    try {
      const response = await axios.get(url);
      const data = response.data;
      const currentWeather = data.current_weather;
      const hourlyWeather = data.hourly;

      // Extract relevant data
      const temperature = currentWeather.temperature;
      const humidity = hourlyWeather.relativehumidity_2m[0];
      const windSpeed = hourlyWeather.windspeed_10m[0];
      const precipitation = hourlyWeather.precipitation[0];

      // Update weather data state
      setWeatherData({
        temperature,
        humidity,
        windSpeed,
        precipitation,
      });
    } catch (error) {
      console.error('Error fetching weather data:', error);
    }
  };

  useEffect(() => {
    fetchWeatherData();
  }, []);

  const selectedAcUnit = acUnits.find((unit) => unit.id === selectedAC);

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Smart AC Control</Text>

      <View style={styles.cardContainer}>
        <Text style={styles.label}>Select AC</Text>
        <Picker selectedValue={selectedAC} onValueChange={(itemValue) => setSelectedAC(itemValue)} style={styles.picker}>
          {acUnits.map((unit) => (
            <Picker.Item key={unit.id} label={unit.name} value={unit.id} />
          ))}
        </Picker>
      </View>

      {selectedAcUnit && (
        <View style={styles.cardContainer}>
          <Text style={styles.label}>AC Control</Text>
          <View style={styles.row}>
            <Text style={styles.temperatureText}>{selectedAcUnit.temp}°C</Text>
            <View>
              <Button icon={<Ionicons name="add" size={24} />} type="clear" onPress={() => adjustTemperature(1)} />
              <Button icon={<Ionicons name="remove" size={24} />} type="clear" onPress={() => adjustTemperature(-1)} />
            </View>
          </View>

          <View style={styles.switchContainer}>
            <Text style={styles.powerText}>{selectedAcUnit.isOn ? "ON" : "OFF"}</Text>
            <Switch value={selectedAcUnit.isOn} onValueChange={togglePower} />
          </View>

          <Picker selectedValue={selectedAcUnit.mode} onValueChange={(itemValue) => changeMode(itemValue)} style={styles.picker}>
            <Picker.Item label="Cool" value="Cool" />
            <Picker.Item label="Heat" value="Heat" />
            <Picker.Item label="Fan" value="Fan" />
          </Picker>
        </View>
      )}

      {/* Real-Time Weather Section with Improved UI */}
      <View style={styles.weatherContainer}>
        <ScrollView horizontal pagingEnabled showsHorizontalScrollIndicator={false} contentContainerStyle={styles.weatherScroll}>
          <View style={styles.weatherCard}>
            <Text style={styles.weatherTitle}>Air Quality Index</Text>
            <Text style={styles.weatherValue}>--</Text>
          </View>
          <View style={styles.weatherCard}>
            <Text style={styles.weatherTitle}>Temperature</Text>
            <Text style={styles.weatherValue}>{weatherData.temperature}°C</Text>
          </View>
          <View style={styles.weatherCard}>
            <Text style={styles.weatherTitle}>Forecast</Text>
            <Text style={styles.weatherValue}>--</Text>
          </View>
          <View style={styles.weatherCard}>
            <Text style={styles.weatherTitle}>Precipitation</Text>
            <Text style={styles.weatherValue}>{weatherData.precipitation}%</Text>
          </View>
          <View style={styles.weatherCard}>
            <Text style={styles.weatherTitle}>Humidity</Text>
            <Text style={styles.weatherValue}>{weatherData.humidity}%</Text>
          </View>
          <View style={styles.weatherCard}>
            <Text style={styles.weatherTitle}>Wind Speed</Text>
            <Text style={styles.weatherValue}>{weatherData.windSpeed} km/h</Text>
          </View>
        </ScrollView>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff", padding: 20 },
  title: { fontSize: 26, fontWeight: "bold", marginBottom: 20, textAlign: "center" },
  cardContainer: { marginBottom: 20, padding: 20, backgroundColor: "#f5f5f5", borderRadius: 10 },
  label: { fontWeight: "bold", fontSize: 18 },
  row: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginVertical: 10 },
  temperatureText: { fontSize: 36, fontWeight: "bold" },
  switchContainer: { flexDirection: "row", alignItems: "center", marginTop: 10 },
  powerText: { fontSize: 18, fontWeight: "bold", marginRight: 10 },
  picker: { height: 50, width: "100%" },
  weatherContainer: { marginTop: 20, height: 140, backgroundColor: "#e0e0e0", borderRadius: 10, overflow: "hidden", padding: 10 },
  weatherScroll: { flexDirection: "row", alignItems: "center" },
  weatherCard: { width: 160, justifyContent: "center", alignItems: "center", padding: 15, backgroundColor: "#fff", borderRadius: 8, marginHorizontal: 5, elevation: 2 },
  weatherTitle: { fontSize: 14, fontWeight: "bold", color: "#333", textAlign: "center" },
  weatherValue: { fontSize: 20, color: "#007AFF", textAlign: "center", marginTop: 5 },
});

export default HomeScreen;