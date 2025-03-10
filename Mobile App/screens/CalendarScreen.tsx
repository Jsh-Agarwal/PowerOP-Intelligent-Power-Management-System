import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, TouchableOpacity, Animated, Easing, ScrollView, Alert } from "react-native";
import { Calendar, DateData } from "react-native-calendars";
import DateTimePicker from "@react-native-community/datetimepicker";
import { useTheme } from "../components/ThemeContext";
import { LineChart } from "react-native-chart-kit";
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import AnalysisModal from './AnalysisModal';

const CalendarScreen: React.FC = () => {
  const { theme } = useTheme();
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [showTimePicker, setShowTimePicker] = useState<boolean>(false);
  const [selectedTime, setSelectedTime] = useState<Date>(new Date());
  const [energyData, setEnergyData] = useState<any[]>([
    { time: "12 AM", value: 0 },
    { time: "12 PM", value: 0 },
  ]);
  const [costData, setCostData] = useState<any[]>([
    { time: "12 AM", value: 0 },
    { time: "12 PM", value: 0 },
  ]);
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [analysisData, setAnalysisData] = useState<any>({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const scaleAnim = new Animated.Value(1);
  
  const systemId = "test_system"; // Replace with your actual system ID

  const handlePressIn = () => {
    Animated.timing(scaleAnim, {
      toValue: 0.9,
      duration: 100,
      easing: Easing.ease,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.timing(scaleAnim, {
      toValue: 1,
      duration: 100,
      easing: Easing.ease,
      useNativeDriver: true,
    }).start();
  };

  const fetchCostAnalysis = async () => {
    if (!selectedDate) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Retrieve the token from AsyncStorage
      const token = await AsyncStorage.getItem('authToken');

      const response = await axios.get(`http://localhost:8000/api/analysis/cost/${systemId}`, {
        headers: {
          Authorization: `Bearer ${token}`, // Use the retrieved token
        },
        params: {
          start_time: new Date(selectedDate).toISOString(),
          end_time: new Date(selectedDate).toISOString(),
        },
      });
      
      const analysis = response.data?.analysis || {};
      
      // Safely set data with fallbacks
      setEnergyData([
        { time: "12 AM", value: analysis.total_energy_kwh || 0 },
        { 
          time: selectedTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), 
          value: analysis.peak_usage_kwh || 0 
        },
      ]);
      
      setCostData([
        { time: "12 AM", value: analysis.total_cost || 0 },
        { 
          time: selectedTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), 
          value: analysis.peak_usage_cost || 0 
        },
      ]);
    } catch (error) {
      console.error('Error fetching cost analysis:', error);
      setError('Failed to fetch cost analysis');
      
      // Set default data on error
      setEnergyData([
        { time: "12 AM", value: 0 },
        { time: " 12 PM", value: 0 },
      ]);
      
      setCostData([
        { time: "12 AM", value: 0 },
        { time: "12 PM", value: 0 },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeData = async () => {
    if (!selectedDate) {
      Alert.alert('Error', 'Please select a date first');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const token = await AsyncStorage.getItem('authToken');

      const requestData = {
        data: [
          {
            timestamp: new Date().toISOString(),
            temperature: 23.5,
            humidity: 50.0,
            power: 1000.0,
            pressure: 101.3,
          },
        ],
      };

      const [anomalyResponse, costResponse, llmResponse] = await Promise.all([
        axios.post(`http://localhost:8000/api/analysis/anomaly/detect/${systemId}`, requestData, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }).catch(err => {
          console.error('Anomaly detection error:', err);
          return { data: { anomalies: [] } };
        }),

        axios.get(`http://localhost:8000/api/analysis/cost/${systemId}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          params: {
            start_time: new Date(selectedDate).toISOString(),
            end_time: new Date(selectedDate).toISOString(),
          },
        }).catch(err => {
          console.error('Cost analysis error:', err);
          return { data: { analysis: {} } };
        }),

        axios.post(`http://localhost:8000/api/analysis/optimize/llm/${systemId}`, {
          query: "Analyze system efficiency",
          context: {
            temperature: 23.5,
            power: 1000.0,
            runtime_hours: 24,
          },
        }, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }).catch(err => {
          console.error('LLM analysis error:', err);
          return { data: { data: { suggestions: [] } } };
        })
      ]);

      const anomalies = anomalyResponse.data?.anomalies || [];
      const analysis = costResponse.data?.analysis || {};
      const suggestions = llmResponse.data?.data?.suggestions || [];

      setAnalysisData({
        anomalies,
        total_energy_kwh: analysis.total_energy_kwh || 0,
        total_cost: analysis.total_cost || 0,
        average_daily_cost: analysis.average_daily_cost || 0,
        peak_usage_kwh: analysis.peak_usage_kwh || 0,
        peak_usage_cost: analysis.peak_usage_cost || 0,
        efficiency_score: analysis.efficiency_score || 0,
        llm_analysis: suggestions,
      });

      setModalVisible(true);
    } catch (error) {
      console.error('Error analyzing data:', error);
      setError('Failed to analyze data');
      Alert.alert('Error', 'Failed to analyze data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (selectedDate) {
      fetchCostAnalysis();
    }
  }, [selectedDate, selectedTime]);

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme === "dark" ? "#1E1E1E" : "#F5F5F5",
      padding: 20,
    },
    sectionTitle: {
      fontSize: 20,
      fontWeight: "bold",
      marginVertical: 10,
      color: theme === "dark" ? "#E0E0E0" : "#333333",
      textAlign: "center",
    },
    button: {
      backgroundColor: theme === "dark" ? "#BB86FC" : "#6200EE",
      padding: 12,
      borderRadius: 8,
      marginVertical: 15,
      alignItems: "center",
    },
    buttonText: {
      color: "#FFFFFF",
      fontSize: 18,
      fontWeight: "bold",
    },
    graphContainer: {
      alignItems: "center",
      marginTop: 20,
      width: "100%",
      height: 250,
    },
    scrollContainer: {
      flexGrow: 1,
    },
    errorText: {
      color: "red",
      textAlign: "center",
      marginVertical: 10,
    },
    loadingText: {
      textAlign: "center",
      marginVertical: 10,
      color: theme === "dark" ? "#E0E0E0" : "#333333",
    },
  });

  const renderCharts = () => {
 if (energyData.length < 2 || costData.length < 2) {
      return <Text style={styles.loadingText}>No data available</Text>;
    }

    return (
      <>
        <View style={styles.graphContainer}>
          <Text style={styles.sectionTitle}>Energy Efficiency</Text>
          <LineChart
            data={{
              labels: energyData.map(item => item.time),
              datasets: [
                {
                  data: energyData.map(item => item.value),
                },
              ],
            }}
            width={300}
            height={220}
            yAxisLabel=""
            yAxisSuffix=" kW"
            chartConfig={{
              backgroundColor: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
              backgroundGradientFrom: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
              backgroundGradientTo: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
              decimalPlaces: 2,
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
              style: {
                borderRadius: 16,
              },
              propsForDots: {
                r: "6",
                strokeWidth: "2",
                stroke: "#BB86FC",
              },
            }}
            bezier
          />
        </View>

        <View style={styles.graphContainer}>
          <Text style={styles.sectionTitle}>Cost Used</Text>
          <LineChart
            data={{
              labels: costData.map(item => item.time),
              datasets: [
                {
                  data: costData.map(item => item.value),
                },
              ],
            }}
            width={300}
            height={220}
            yAxisLabel=""
            yAxisSuffix=" $"
            chartConfig={{
              backgroundColor: "#1E1E1E",
              backgroundGradientFrom: "#1E1E1E",
              backgroundGradientTo: "#1E1E1E",
              decimalPlaces: 2,
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
              style: {
                borderRadius: 16,
              },
              propsForDots: {
                r: "6",
                strokeWidth: "2",
                stroke: "#BB86FC",
              },
            }}
            bezier
          />
        </View>
      </>
    );
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        <Calendar
          onDayPress={(day: DateData) => setSelectedDate(day.dateString)}
          markedDates={{ [selectedDate]: { selected: true, selectedColor: "#6200EE" } }}
          theme={{
            backgroundColor: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
            calendarBackground: theme === "dark" ? "#1E1E1E" : "#FFFFFF",
            dayTextColor: theme === "dark" ? "#E0E0E0" : "#333333",
            todayTextColor: theme === "dark" ? "#BB86FC" : "#6200EE",
            selectedDayBackgroundColor: "#6200EE",
            monthTextColor: theme === "dark" ? "#E0E0E0" : "#333333",
          }}
        />

        <Text style={styles.sectionTitle}>Selected Date: {selectedDate || "None"}</Text>

        <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
          <TouchableOpacity 
            onPress={() => setShowTimePicker(true)} 
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            style={styles.button}
          >
            <Text style={styles.buttonText}>Set Time</Text>
          </TouchableOpacity>
        </Animated.View>

        {showTimePicker && (
          <DateTimePicker
            value={selectedTime}
            mode="time"
            is24Hour={true}
            display="default"
            onChange={(event, date) => {
              setShowTimePicker(false);
              if (date) setSelectedTime(date);
            }}
          />
        )}

        {error && <Text style={styles.errorText}>{error}</Text>}
        {isLoading ? (
          <Text style={styles.loadingText}>Loading data...</Text>
        ) : (
          renderCharts()
        )}

        <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
          <TouchableOpacity 
            onPress={analyzeData}
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            style={styles.button}
          >
            <Text style={styles.buttonText}>Analyze My Data</Text>
          </TouchableOpacity>
        </Animated.View>

        <AnalysisModal 
          visible={modalVisible} 
          onClose={() => setModalVisible(false)} 
          analysisData={analysisData} 
        />
      </View>
    </ScrollView>
  );
};

export default CalendarScreen;