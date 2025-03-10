import React from "react"
import { Pressable, StyleSheet, Text as RNText, View, Switch } from "react-native"

import { ReactNode } from "react";

interface ButtonProps {
  children: ReactNode;
  onPress: () => void;
  style?: object;
}

export const Button = ({ children, onPress, style }: ButtonProps) => (
  <Pressable onPress={onPress} style={[styles.button, style]}>
    <RNText style={styles.buttonText}>{children}</RNText>
  </Pressable>
)

interface CustomTextProps {
  children: ReactNode;
  style?: object;
  variant?: "h6" | "h4" | "text";
}

export const CustomText = ({ children, style, variant }: CustomTextProps) => {
  let textStyle = styles.text
  if (variant === "h6") textStyle = styles.h6
  if (variant === "h4") textStyle = styles.h4
  return <RNText style={[textStyle, style]}>{children}</RNText>
}

interface InputProps {
  style?: object;
}

export const Input = ({ style }: InputProps) => <View style={[styles.input, style]} />

interface CardProps {
  children: ReactNode;
  style?: object;
}

export const Card = ({ children, style }: CardProps) => <View style={[styles.card, style]}>{children}</View>

interface SelectProps {
  value: any;
  onValueChange: (value: any) => void;
  children: ReactNode;
}

export const Select = ({ value, onValueChange, children }: SelectProps) => (
  <View>{React.Children.map(children, (child) => React.isValidElement(child) ? React.cloneElement(child, { selectedValue: value, onValueChange } as any) : child)}</View>
)

interface SelectItemProps {
  label: string;
  value: any;
  selectedValue: any;
  onValueChange: (value: any) => void;
}

export const SelectItem = ({ label, value, selectedValue, onValueChange }: SelectItemProps) => (
  <Pressable onPress={() => onValueChange(value)}>
    <RNText>{label}</RNText>
  </Pressable>
)

interface SwitchComponentProps {
  value: boolean;
  onValueChange: (value: boolean) => void;
}

export const SwitchComponent = ({ value, onValueChange }: SwitchComponentProps) => <Switch value={value} onValueChange={onValueChange} />

interface TableProps {
  children: ReactNode;
}

export const Table = ({ children }: TableProps) => <View>{children}</View>
interface TableHeaderProps {
  children: ReactNode;
}

export const TableHeader = ({ children }: TableHeaderProps) => <View>{children}</View>
interface TableRowProps {
  children: ReactNode;
}

export const TableRow = ({ children }: TableRowProps) => <View style={styles.tableRow}>{children}</View>
interface TableCellProps {
  children: ReactNode;
}

export const TableCell = ({ children }: TableCellProps) => <View style={styles.tableCell}>{children}</View>

const styles = StyleSheet.create({
  button: {
    backgroundColor: "#6200EE",
    padding: 10,
    borderRadius: 5,
    marginVertical: 5,
  },
  buttonText: {
    color: "#FFFFFF",
    textAlign: "center",
  },
  text: {
    fontSize: 16,
  },
  h6: {
    fontSize: 18,
    fontWeight: "bold",
  },
  h4: {
    fontSize: 24,
    fontWeight: "bold",
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 10,
    borderRadius: 5,
    marginBottom: 10,
  },
  card: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 10,
    borderRadius: 5,
    marginBottom: 10,
  },
  tableRow: {
    flexDirection: "row",
  },
  tableCell: {
    flex: 1,
    padding: 10,
    borderRightWidth: 1,
    borderColor: "#ccc",
  },
})

