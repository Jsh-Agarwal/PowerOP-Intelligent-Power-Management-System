import React, { useState, useEffect, useRef } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Animated, Dimensions, SafeAreaView } from "react-native";
import { useTheme } from "../components/ThemeContext";
import { MaterialIcons } from "@expo/vector-icons";
import { LineChart } from "react-native-chart-kit";
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

type ChartType = "Room Temp Over Time" | "Outdoor vs. Indoor Temp" | "Cost Saved";

const chartTypes: ChartType[] = ["Room Temp Over Time", "Outdoor vs. Indoor Temp", "Cost Saved"];
const screenWidth = Dimensions.get("window").width - 40; // Account for padding

// Your FastAPI base URL - keep this as localhost:8000
const LOCAL_API_BASE_URL = 'http://localhost:8000';
// Replace with your actual device and zone IDs
const DEVICE_ID = "test_device";
const ZONE_ID = "main";

const StatsScreen: React.FC = () => {
  const { theme } = useTheme(); // Assuming useTheme provides a theme
  const [selectedTimeFrame, setSelectedTimeFrame] = useState("Weekly");
  const [comfortLevel, setComfortLevel] = useState(50);
  const comfortAnim = useRef(new Animated.Value(50)).current;
  const [outdoorTemp, setOutdoorTemp] = useState<number>(0);
  const [indoorTemp, setIndoorTemp] = useState<number>(0);
  const [temperatureHistory, setTemperatureHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [authToken, setAuthToken] = useState<string | null>(null);

  // Get auth token from AsyncStorage
  useEffect(() => {
    const getAuthToken = async () => {
      try {
        const token = await AsyncStorage.getItem('authToken');
        setAuthToken(token);
      } catch (error) {
        console.error('Error retrieving auth token:', error);
      }
    };
    
    getAuthToken();
  }, []);

  useEffect(() => {
    Animated.timing(comfortAnim, {
      toValue: comfortLevel,
      duration: 500,
      useNativeDriver: false,
    }).start();
  }, [comfortLevel]);

  useEffect(() => {
    // Only fetch data if we have an auth token
    if (authToken) {
      const fetchData = async () => {
        try {
          setIsLoading(true);
          await Promise.all([
            fetchOutdoorTemperature(),
            fetchIndoorTemperature(),
            fetchIndoorTemperatureHistory(),
          ]);
        } catch (error) {
          console.error('Error fetching data:', error);
        } finally {
          setIsLoading(false);
        }
      };

      fetchData();
    }
  }, [authToken]); // Run when authToken changes

  const fetchOutdoorTemperature = async () => {
    const latitude = '28.7041'; // Example latitude for New Delhi
    const longitude = '77.1025'; // Example longitude for New Delhi
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`;

    try {
      const response = await axios.get(url);
      const currentWeather = response.data.current_weather;
      setOutdoorTemp(currentWeather.temperature);
      return true;
    } catch (error) {
      console.error('Error fetching outdoor temperature:', error);
      return false;
    }
  };

  const fetchIndoorTemperature = async () => {
    if (!authToken) return false;
    
    try {
      const response = await axios.get(`${LOCAL_API_BASE_URL}/api/temperature/current`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        params: {
          device_id: DEVICE_ID,
          zone_id: ZONE_ID,
        },
      });
      setIndoorTemp(response.data.temperature);
      return true;
    } catch (error) {
      console.error('Error fetching indoor temperature:', error);
      // Don't set fallback values - keep what was there before
      return false;
    }
  };

  const fetchIndoorTemperatureHistory = async () => {
    if (!authToken) return false;
    
    try {
      const response = await axios.get(`${LOCAL_API_BASE_URL}/api/temperature/history`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        params: {
          device_id: DEVICE_ID,
          zone_id: ZONE_ID,
          start_time: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), // Last 24 hours
          end_time: new Date().toISOString(),
        },
      });
      
      if (response.data && response.data.history) {
        setTemperatureHistory(response.data.history);
      } else {
        console.warn('Invalid temperature history data format');
      }
      return true;
    } catch (error) {
      console.error('Error fetching indoor temperature history:', error);
      return false;
    }
  };

  const getMoodIcon = () => {
    if (comfortLevel < 30) return "sentiment-very-dissatisfied";
    if (comfortLevel < 70) return "sentiment-neutral";
    return "sentiment-very-satisfied";
  };

  const getChartData = (type: ChartType) => {
    switch (type) {
      case "Room Temp Over Time": {
        // Make sure we have valid temperature history data
        if (!temperatureHistory || temperatureHistory.length === 0) {
          return {
            labels: ["No Data"],
            datasets: [{ data: [0], color: () => "rgba(255, 99, 132, 1)", strokeWidth: 2 }],
            legend: ["Temperature"]
          };
        }
        
        return {
          labels: temperatureHistory.map(item => 
            new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          ),
          datasets: [
            {
              data: temperatureHistory.map(item => item.temperature),
              color: (opacity = 1) => `rgba(255, 99, 132, ${opacity})`,
              strokeWidth: 2
            }
          ],
          legend: ["Temperature"]
        };
      }
      
      case "Outdoor vs. Indoor Temp": {
        return {
          labels: ["Now"],
          datasets: [
            {
              data: [outdoorTemp],
              color: (opacity = 1) => `rgba(255, 99, 132, ${opacity})`,
              strokeWidth: 2
            },
            {
              data: [indoorTemp],
              color: (opacity = 1) => `rgba(54, 162, 235, ${opacity})`,
              strokeWidth: 2
            }
          ],
          legend: ["Outdoor", "Indoor"]
        };
      }
      
      case "Cost Saved":
      default:
        return {
          labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
          datasets: [
            {
              data: [2000, 4500, 2800, 8000, 9900, 4300],
              color: (opacity = 1) => `rgba(255, 99, 132, ${opacity})`,
              strokeWidth: 2
            }
          ],
          legend: ["Savings"]
        };
    }
  };

  const chartConfig = {
    backgroundGradientFrom: "#fff",
    backgroundGradientTo: "#fff",
    decimalPlaces: 1,
    color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
    style: {
      borderRadius: 16
    },
    propsForDots: {
      r: "6",
      strokeWidth: "2",
      stroke: "#ffa726"
    }
  };

  // Safe rendering function for charts
  const renderChart = (title: ChartType) => {
    try {
      const chartData = getChartData(title);
      
      // Check if chart data is valid
      if (!chartData || !chartData.datasets || chartData.datasets.length === 0) {
        return (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>Unable to load chart data</Text>
          </View>
        );
      }
      
      return (
        <View key={title} style={styles.chartContainer}>
          <Text style={styles.sectionTitle}>{title}</Text>
          <LineChart
            data={chartData}
            width={screenWidth}
            height={250}
            chartConfig={chartConfig}
            bezier
            style={styles.chart}
            fromZero
            yAxisLabel={title === "Cost Saved" ? "₹" : ""}
            yAxisSuffix={title === "Cost Saved" ? "" : "°C"}
            withInnerLines={true}
            withOuterLines={true}
            withDots={true}
            withShadow={false}
            withHorizontalLabels={true}
            withVerticalLabels={true}
            withHorizontalLines={true}
            withVerticalLines={false}
          />
          {/* Display legend manually */}
          {chartData.legend && (
            <View style={styles.legendContainer}>
              {chartData.legend.map((item, index) => (
                <View key={index} style={styles.legendItem}>
                  <View
                    style={[
                      styles.legendColor,
                      {
                        backgroundColor: 
                          index === 0 
                            ? "rgb(255, 99, 132)" 
                            : "rgb(54, 162, 235)"
                      }
                    ]}
                  />
                  <Text style={styles.legendText}>{item}</Text>
                </View>
              ))}
            </View>
          )}
        </View>
      );
    } catch (error) {
      console.error(`Error rendering chart ${title}:`, error);
      return (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>Chart unavailable</Text>
        </View>
      );
    }
  };

  // Show a loading or unauthorized state if no token
  if (!authToken) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={[styles.container, styles.centerContent]}>
          <Text style={styles.sectionTitle}>Not authenticated</Text>
          <Text>Please log in to view statistics</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView style={styles.container}>
        {/* Timeframe Selection */}
        <View style={styles.toggleContainer}>
          {["Weekly", "Monthly", "Yearly"].map((option) => (
            <TouchableOpacity
              key={option}
              style={[styles.toggleButton, selectedTimeFrame === option && styles.selectedButton]}
              onPress={() => setSelectedTimeFrame(option)}
            >
              <Text style={styles.toggleText}>{option}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Temperature Boxes */}
        <View style={styles.tempBoxContainer}>
          <View style={styles.tempBox}>
            <Text>🌤️ Outdoor Temp: {isNaN(outdoorTemp) ? "--" : outdoorTemp.toFixed(1)}°C</Text>
          </View>
          <View style={styles.tempBox}>
            <Text>🏠 Indoor Temp: {isNaN(indoorTemp) ? "--" : indoorTemp.toFixed(1)}°C</Text>
          </View>
        </View>

        {/* Charts */}
        {chartTypes.map(type => renderChart(type))}

        {/* Comfort Meter */}
        <Text style={styles.sectionTitle}>Comfort Meter</Text>
        <View style={styles.thermometerContainer}>
          <View style={styles.thermometer}>
            <Animated.View
              style={[
                styles.thermometerFill,
                {
                  height: comfortAnim.interpolate({
                    inputRange: [0, 100],
                    outputRange: [0, 200],
                  }),
                },
              ]}
            />
          </View>
          <MaterialIcons
            name={getMoodIcon()}
            size={40}
            color="#ff5733"
            style={styles.moodIcon}
          />
          <Text style={{ fontSize: 18, fontWeight: "bold", color: "#ff5733" }}>
            Comfort Level: {Math.round(comfortLevel)}
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  container: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    padding: 20,
  },
  centerContent: {
    justifyContent: "center",
    alignItems: "center",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
    color: "#000",
  },
  chartContainer: {
    marginBottom: 20,
    width: "100%",
    alignItems: "center",
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  toggleContainer: {
    flexDirection: "row",
    justifyContent: "center",
    marginBottom: 10,
  },
  toggleButton: {
    padding: 10,
    marginHorizontal: 5,
    borderRadius: 10,
    backgroundColor: "#36A2EB",
  },
  selectedButton: {
    backgroundColor: "#0056b3",
  },
  toggleText: {
    color: "#FFF",
    fontWeight: "bold",
  },
  tempBoxContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 10,
  },
  tempBox: {
    backgroundColor: "#EEE",
    padding: 10,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 5,
    alignItems: "center",
  },
  thermometerContainer: {
    alignItems: "center",
    marginVertical: 20,
  },
  thermometer: {
    width: 50,
    height: 200,
    borderRadius: 25,
    backgroundColor: "#ddd",
    overflow: "hidden",
    borderWidth: 2,
    borderColor: "#333",
  },
  thermometerFill: {
    position: "absolute",
    bottom: 0,
    width: "100%",
    backgroundColor: "#ff5733",
  },
  moodIcon: {
    fontSize: 40,
    marginTop: 10,
  },
  legendContainer: {
    flexDirection: "row",
    justifyContent: "center",
    marginTop: 10,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    marginHorizontal: 10,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 5,
  },
  legendText: {
    fontSize: 12,
    color: "#333",
  },
  errorContainer: {
    height: 220,
    width: screenWidth,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f8f8f8",
    borderRadius: 16,
    marginVertical: 8,
  },
  errorText: {
    color: "#666",
    fontSize: 14,
  }
});

export default StatsScreen;