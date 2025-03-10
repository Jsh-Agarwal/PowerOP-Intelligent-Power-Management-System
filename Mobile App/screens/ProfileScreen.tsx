import React from "react";
import { View, StyleSheet, ScrollView, Text as RNText, Switch as RNSwitch, TouchableOpacity } from "react-native";
import { useTheme } from "../components/ThemeContext";
import type { StackNavigationProp } from "@react-navigation/stack";

// Define the navigation stack
type RootStackParamList = {
  Profile: undefined;
  Login: undefined;
};

type ProfileScreenNavigationProp = StackNavigationProp<RootStackParamList, "Profile">;

interface ProfileScreenProps {
  navigation: ProfileScreenNavigationProp;
}

// ProfileScreen Component
const ProfileScreen: React.FC<ProfileScreenProps> = ({ navigation }) => {
  const { theme, toggleTheme } = useTheme();

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme === "dark" ? "#121212" : "#F4F4F4",
      padding: 20,
    },
    button: {
      marginTop: 10,
      paddingVertical: 12,
      paddingHorizontal: 20,
      borderRadius: 30,
      backgroundColor: "#6200EE",
      alignItems: "center",
      justifyContent: "center",
    },
    buttonText: {
      color: "#FFFFFF",
      fontSize: 16,
      fontWeight: "bold",
    },
    card: {
      padding: 20,
      borderRadius: 15,
      backgroundColor: theme === "dark" ? "#333" : "#FFFFFF",
      shadowColor: "#000",
      shadowOpacity: 0.2,
      shadowRadius: 8,
      elevation: 5,
    },
    headerText: {
      fontSize: 20,
      fontWeight: "bold",
      color: theme === "dark" ? "#FFF" : "#333",
      marginBottom: 10,
    },
    rowText: {
      fontSize: 16,
      color: theme === "dark" ? "#BBB" : "#333",
    },
  });

  return (
    <ScrollView style={styles.container}>
      {/* User Information Card */}
      <View style={styles.card}>
        <RNText style={styles.headerText}>User Information</RNText>
        <RNText>Name: John Doe</RNText>
        <RNText>Email: john.doe@example.com</RNText>
        <TouchableOpacity style={styles.button} onPress={() => console.log("Edit Profile")}>
          <RNText style={styles.buttonText}>Edit Profile</RNText>
        </TouchableOpacity>
      </View>

      {/* Settings Card */}
      <View style={styles.card}>
        <RNText style={styles.headerText}>Settings</RNText>
        <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <RNText style={styles.rowText}>Dark Mode</RNText>
          <RNSwitch value={theme === "dark"} onValueChange={toggleTheme} />
        </View>
      </View>

      {/* Log Out Button */}
      <TouchableOpacity style={styles.button} onPress={() => navigation.navigate("Login")}>
        <RNText style={styles.buttonText}>Log Out</RNText>
      </TouchableOpacity>
    </ScrollView>
  );
};

export default ProfileScreen;
