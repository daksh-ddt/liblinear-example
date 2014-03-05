package com.gavinmhackeling;

public class Counter {
	
	int value = 1;
	
	public void increment() {
		++value; 
	}
	
	public int getValue() {
		return value;
	}

	@Override
	public String toString() {
		return Integer.toString(value);
	}

}
