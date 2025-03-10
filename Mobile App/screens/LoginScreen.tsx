import React, { useState } from "react";
import { View, StyleSheet, TouchableOpacity, TextInput, Alert } from "react-native";
import { Text } from "react-native-elements";
import { useTheme } from "../components/ThemeContext";
import { StackNavigationProp } from '@react-navigation/stack';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

type RootStackParamList = {
  LoginScreen: undefined;
  AdminApp: undefined;
  UserApp: undefined;
};

type LoginScreenNavigationProp = StackNavigationProp<RootStackParamList, 'LoginScreen'>;

interface Props {
  navigation: LoginScreenNavigationProp;
}

const LoginScreen = ({ navigation }: Props) => {
  const { theme } = useTheme();
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");

  const handleLogin = async (userType: "admin" | "user") => {
    try {
      // Send POST request to the /token endpoint
      const response = await axios.post('http://localhost:8000/token', {
        username,
        password,
      });

      // Extract the access token from the response
      const { access_token } = response.data;

      // Store the token in AsyncStorage
      await AsyncStorage.setItem('authToken', access_token);

      // Navigate to the appropriate app
      navigation.navigate(userType === "admin" ? "AdminApp" : "UserApp");;
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert('Login Failed', 'Invalid username or password.');
    }
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      justifyContent: "center",
      alignItems: "center",
      backgroundColor: theme === "dark" ? "#121212" : "#FFFFFF",
      padding: 20,
    },
    title: {
      fontSize: 24,
      marginBottom: 20,
      fontWeight: "bold",
      color: theme === "dark" ? "#FFFFFF" : "#000000",
    },
    input: {
      width: '100%',
      padding: 10,
      marginVertical: 10,
      borderWidth: 1,
      borderColor: theme === "dark" ? "#FFFFFF" : "#000000",
      borderRadius: 5,
      backgroundColor: theme === "dark" ? "#333333" : "#DDDDDD",
      color: theme === "dark" ? "#FFFFFF" : "#000000",
    },
    boxContainer: {
      flexDirection: "row",
      justifyContent: "space-between",
      width: "100%",
      paddingHorizontal: 10,
    },
    box: {
      flex: 1,
      backgroundColor: theme === "dark" ? "#333333" : "#DDDDDD",
      padding: 30,
      margin: 10,
      borderRadius: 10,
      alignItems: "center",
      justifyContent: "center",
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.2,
      shadowRadius: 4,
      elevation: 5, // For Android shadow effect
    },
    boxText: {
      fontSize: 18,
      fontWeight: "bold",
      color: theme === "dark" ? "#FFFFFF" : "#000000",
    },
  });

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Login</Text>
      <TextInput
        style={styles.input}
        placeholder="Username"
        placeholderTextColor={theme === "dark" ? "#FFFFFF" : "#000000"}
        value={username}
        onChangeText={setUsername}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor={theme === "dark" ? "#FFFFFF" : "#000000"}
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <View style={styles.boxContainer}>
        <TouchableOpacity style={styles.box} onPress={() => handleLogin("admin")}>
          <Text style={styles.boxText}>Admin Login</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.box} onPress={() => handleLogin("user")}>
          <Text style={styles.boxText}>User  Login</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

export default LoginScreen;