
import React, { createContext, useContext, ReactNode, useState, useEffect, useCallback } from 'react';
import type {
    Product, ProductBatch, SaleInvoice, PurchaseInvoice, PurchaseInvoiceItem, InvoiceItem,
    Customer, Supplier, Employee, Expense, Service, StoreSettings, CartItem,
    CustomerTransaction, SupplierTransaction, PayrollTransaction, ActivityLog,
    User, Role, Permission, AppState
} from './types';
import { formatCurrency } from './utils/formatters';
import { api } from './services/supabaseService';
import { supabase } from './utils/supabaseClient';

interface AppContextType extends AppState {
    showToast: (message: string) => void;
    isLoading: boolean;
    isLoggingOut: boolean;
    
    // Auth
    login: (identifier: string, password: string, type: 'admin' | 'staff') => Promise<{ success: boolean; message: string; pending?: boolean; locked?: boolean }>;
    signup: (email: string, password: string) => Promise<{ success: boolean; message: string }>;
    logout: () => Promise<{ success: boolean; message: string }>;
    hasPermission: (permission: Permission) => boolean;
    
    // Backup & Restore
    exportData: () => void;
    importData: (file: File) => void;
    cloudBackup: () => Promise<boolean>;
    cloudRestore: () => Promise<boolean>;
    autoBackupEnabled: boolean;
    setAutoBackupEnabled: (enabled: boolean) => void;

    // Users & Roles
    addUser: (user: Omit<User, 'id'>) => Promise<{ success: boolean; message: string }>;
    updateUser: (user: Partial<User> & { id: string }) => Promise<{ success: boolean; message: string }>;
    deleteUser: (userId: string) => Promise<void>;
    addRole: (role: Omit<Role, 'id'>) => Promise<{ success: boolean; message: string }>;
    updateRole: (role: Role) => Promise<{ success: boolean; message: string }>;
    deleteRole: (roleId: string) => Promise<void>;

    // Inventory Actions
    addProduct: (product: Omit<Product, 'id' | 'batches'>, firstBatch: Omit<ProductBatch, 'id'>) => { success: boolean; message: string }; 
    updateProduct: (product: Product) => { success: boolean; message: string };
    deleteProduct: (productId: string) => void;
    
    // POS Actions
    addToCart: (itemToAdd: Product | Service, type: 'product' | 'service') => { success: boolean; message: string };
    updateCartItemQuantity: (itemId: string, itemType: 'product' | 'service', newQuantity: number) => { success: boolean; message: string };
    updateCartItemFinalPrice: (itemId: string, itemType: 'product' | 'service', finalPrice: number) => void;
    removeFromCart: (itemId: string, itemType: 'product' | 'service') => void;
    completeSale: (cashier: string, customerId?: string) => Promise<{ success: boolean; invoice?: SaleInvoice; message: string }>;
    beginEditSale: (invoiceId: string) => { success: boolean; message: string; customerId?: string; };
    cancelEditSale: () => void;
    addSaleReturn: (originalInvoiceId: string, returnItems: { id: string; type: 'product' | 'service'; quantity: number }[], cashier: string) => { success: boolean, message: string };
    setInvoiceTransientCustomer: (invoiceId: string, customerName: string) => Promise<void>;
    
    // Purchase Actions
    addPurchaseInvoice: (invoiceData: Omit<PurchaseInvoice, 'id' | 'totalAmount' | 'items' | 'type' | 'originalInvoiceId'> & { items: Omit<PurchaseInvoiceItem, 'productName'>[] }) => { success: boolean, message: string };
    beginEditPurchase: (invoiceId: string) => { success: boolean; message: string };
    cancelEditPurchase: () => void;
    updatePurchaseInvoice: (invoiceData: Omit<PurchaseInvoice, 'id' | 'totalAmount' | 'items' | 'type' | 'originalInvoiceId'> & { items: Omit<PurchaseInvoiceItem, 'productName'>[] }) => { success: boolean, message: string };
    addPurchaseReturn: (originalInvoiceId: string, returnItems: { productId: string; lotNumber: string, quantity: number }[]) => { success: boolean; message: string };

    // Settings
    updateSettings: (newSettings: StoreSettings) => void;
    
    // Services
    addService: (service: Omit<Service, 'id'>) => void;
    deleteService: (serviceId: string) => void;
    
    // Accounting
    addSupplier: (supplier: Omit<Supplier, 'id' | 'balance'>, initialBalance?: { amount: number, type: 'creditor' | 'debtor', currency: 'AFN' | 'USD', exchangeRate?: number }) => void;
    deleteSupplier: (id: string) => void;
    addSupplierPayment: (supplierId: string, amount: number, description: string, currency?: 'AFN' | 'USD', exchangeRate?: number) => SupplierTransaction;
    
    addCustomer: (customer: Omit<Customer, 'id' | 'balance'>, initialBalance?: { amount: number, type: 'creditor' | 'debtor', currency: 'AFN' | 'USD', exchangeRate?: number }) => void;
    deleteCustomer: (id: string) => void;
    addCustomerPayment: (customerId: string, amount: number, description: string) => CustomerTransaction;
    
    addEmployee: (employee: Omit<Employee, 'id'|'balance'>) => void;
    addEmployeeAdvance: (employeeId: string, amount: number) => void;
    processAndPaySalaries: () => { success: boolean; message: string };
    addExpense: (expense: Omit<Expense, 'id'>) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

const getDeviceId = () => {
    let id = localStorage.getItem('ketabestan_device_id');
    if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem('ketabestan_device_id', id);
    }
    return id;
};

const getDefaultState = (): AppState => {
    return {
        products: [], saleInvoices: [], purchaseInvoices: [], customers: [],
        suppliers: [], employees: [], expenses: [], services: [],
        storeSettings: {
            storeName: 'کتابستان', address: '', phone: '', lowStockThreshold: 10,
            expiryThresholdMonths: 3, currencyName: 'افغانی', currencySymbol: 'AFN'
        },
        cart: [], customerTransactions: [], supplierTransactions: [], payrollTransactions: [],
        activities: [], saleInvoiceCounter: 0, editingSaleInvoiceId: null, editingPurchaseInvoiceId: null,
        isAuthenticated: false, currentUser: null,
        users: [],
        roles: [],
    };
};

const generateNextId = (prefix: string, ids: string[]): string => {
    let max = 0;
    const regex = new RegExp(`^${prefix}(\\d+)$`); 
    for (const id of ids) {
        const match = id.match(regex);
        if (match) {
             const num = parseInt(match[1], 10);
             if (!isNaN(num)) {
                 if (num > max) max = num;
             }
        }
    }
    return `${prefix}${max + 1}`;
};

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [state, setState] = useState<AppState>(getDefaultState());
    const [isLoading, setIsLoading] = useState(true);
    const [isLoggingOut, setIsLoggingOut] = useState(false);
    const [toastMessage, setToastMessage] = useState('');
    const [autoBackupEnabled, setAutoBackupEnabled] = useState(() => localStorage.getItem('ketabestan_auto_backup') === 'true');

    const showToast = useCallback((message: string) => setToastMessage(message), []);

    const fetchData = useCallback(async () => {
        setIsLoading(true);
        try {
            const [settings, users, roles, products, services, entities, transactions, invoices, activity] = await Promise.all([
                api.getSettings().catch(() => ({})),
                api.getUsers().catch(() => []),
                api.getRoles().catch(() => []),
                api.getProducts().catch(() => []),
                api.getServices().catch(() => []),
                api.getEntities().catch(() => ({ customers: [], suppliers: [], employees: [], expenses: [] })),
                api.getTransactions().catch(() => ({ customerTransactions: [], supplierTransactions: [], payrollTransactions: [] })),
                api.getInvoices().catch(() => ({ saleInvoices: [], purchaseInvoices: [] })),
                api.getActivities().catch(() => [])
            ]);

            const { data: { session } } = await supabase.auth.getSession();
            let isAuth = false;
            let restoredUser = null;

            if (session?.user) {
                const profile = await api.getProfile(session.user.id);
                const deviceId = getDeviceId();
                
                if (profile && profile.is_approved) {
                    if (!profile.current_device_id || profile.current_device_id === deviceId) {
                        isAuth = true;
                        restoredUser = { id: session.user.id, username: session.user.email || 'Admin', roleId: 'admin-role' };
                        
                        if (!profile.current_device_id) {
                            const success = await api.updateProfile(session.user.id, { current_device_id: deviceId });
                            if (!success) { isAuth = false; await logout(); }
                        }
                    }
                } else if (!navigator.onLine && localStorage.getItem('ketabestan_offline_auth') === 'true') {
                    isAuth = true;
                    restoredUser = { id: session.user.id, username: session.user.email || 'Admin', roleId: 'admin-role' };
                }
            } else {
                const localStaff = localStorage.getItem('ketabestan_staff_user');
                if (localStaff) {
                    try {
                        const parsedStaff = JSON.parse(localStaff) as User;
                        const dbUser = users.find(u => u.id === parsedStaff.id);
                        if (dbUser) { isAuth = true; restoredUser = dbUser; }
                        else { localStorage.removeItem('ketabestan_staff_user'); }
                    } catch(e) { localStorage.removeItem('ketabestan_staff_user'); }
                }
            }

            setState(prev => ({
                ...prev,
                storeSettings: settings as StoreSettings || prev.storeSettings,
                users,
                roles: roles.length > 0 ? roles : [{ id: 'admin-role', name: 'Admin', permissions: ['page:dashboard', 'page:inventory', 'page:pos', 'page:purchases', 'page:accounting', 'page:reports', 'page:settings', 'inventory:add_product', 'inventory:edit_product', 'inventory:delete_product', 'pos:create_invoice', 'pos:edit_invoice', 'pos:apply_discount', 'pos:create_credit_sale', 'purchase:create_invoice', 'purchase:edit_invoice', 'accounting:manage_suppliers', 'accounting:manage_customers', 'accounting:manage_payroll', 'accounting:manage_expenses', 'settings:manage_store', 'settings:manage_users', 'settings:manage_backup', 'settings:manage_services', 'settings:manage_alerts'] }],
                products, services, customers: entities.customers, suppliers: entities.suppliers,
                employees: entities.employees, expenses: entities.expenses,
                customerTransactions: transactions.customerTransactions,
                supplierTransactions: transactions.supplierTransactions,
                payrollTransactions: transactions.payrollTransactions,
                saleInvoices: invoices.saleInvoices, 
                purchaseInvoices: invoices.purchaseInvoices,
                activities: activity,
                saleInvoiceCounter: invoices.saleInvoices.length,
                isAuthenticated: isAuth,
                currentUser: restoredUser
            }));
        } catch (error) {
            console.error("Error fetching data:", error);
            showToast("⚠️ خطا در دریافت اطلاعات.");
        } finally {
            setIsLoading(false);
        }
    }, [showToast]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // --- Automatic Backup Logic ---
    useEffect(() => {
        if (!state.isAuthenticated || !autoBackupEnabled) return;

        const checkBackup = async () => {
            const lastBackup = localStorage.getItem('ketabestan_last_backup_time');
            const now = Date.now();
            const oneDay = 24 * 60 * 60 * 1000;

            if (!lastBackup || (now - parseInt(lastBackup)) >= oneDay) {
                console.log("Starting automatic 24h backup...");
                
                // 1. Always download locally (as requested)
                exportData();
                
                // 2. If online and admin, save to cloud
                if (navigator.onLine && state.currentUser && state.currentUser.roleId === 'admin-role') {
                    await cloudBackup();
                }

                localStorage.setItem('ketabestan_last_backup_time', now.toString());
                showToast("✅ پشتیبان‌گیری خودکار ۲۴ ساعته انجام شد.");
            }
        };

        const timer = setTimeout(checkBackup, 5000); // Wait 5s after login to avoid heavy initial load
        return () => clearTimeout(timer);
    }, [state.isAuthenticated, autoBackupEnabled, state.currentUser]);

    const login = async (identifier: string, password: string, type: 'admin' | 'staff'): Promise<{ success: boolean; message: string; pending?: boolean; locked?: boolean }> => {
        if (type === 'admin') {
            try {
                const { data, error } = await supabase.auth.signInWithPassword({ email: identifier, password });
                if (error) return { success: false, message: 'ایمیل یا رمز عبور اشتباه است.' };
                const profile = await api.getProfile(data.user.id);
                if (!profile) return { success: false, message: 'پروفایل یافت نشد.' };
                if (!profile.is_approved) return { success: false, message: 'حساب در انتظار تایید است.', pending: true };
                const deviceId = getDeviceId();
                if (profile.current_device_id && profile.current_device_id !== deviceId) return { success: false, message: 'این حساب در دستگاه دیگری فعال است.', locked: true };
                if (!profile.current_device_id) await api.updateProfile(data.user.id, { current_device_id: deviceId });
                localStorage.setItem('ketabestan_offline_auth', 'true');
                await fetchData();
                return { success: true, message: '✅ ورود موفق' };
            } catch (e) { return { success: false, message: '❌ خطا در اتصال.' }; }
        } else {
            const user = await api.verifyStaffCredentials(identifier, password);
            if (user) {
                localStorage.setItem('ketabestan_staff_user', JSON.stringify(user));
                await fetchData();
                return { success: true, message: `✅ خوش آمدید ${user.username}` };
            } else return { success: false, message: 'نام کاربری یا رمز عبور اشتباه است.' };
        }
    };

    const signup = async (email: string, password: string): Promise<{ success: boolean; message: string }> => {
        try {
            const { error } = await supabase.auth.signUp({ email, password });
            if (error) return { success: false, message: error.message };
            return { success: true, message: '✅ ثبت‌نام انجام شد.' };
        } catch (e) { return { success: false, message: '❌ خطا در ثبت‌نام.' }; }
    };

    const logout = async (): Promise<{ success: boolean; message: string }> => {
        setIsLoggingOut(true);
        const localStaff = localStorage.getItem('ketabestan_staff_user');
        if (localStaff) {
            localStorage.removeItem('ketabestan_staff_user');
            setTimeout(() => {
                setState(prev => ({ ...prev, isAuthenticated: false, currentUser: null }));
                setIsLoggingOut(false);
            }, 800);
            return { success: true, message: 'خروج موفق' };
        }

        if (!navigator.onLine) {
            setIsLoggingOut(false);
            showToast("⚠️ خروج مدیر نیاز به اینترنت دارد.");
            return { success: false, message: 'عدم اتصال' };
        }

        try {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) await api.updateProfile(user.id, { current_device_id: null });
            await supabase.auth.signOut();
            localStorage.removeItem('ketabestan_offline_auth');
            setTimeout(() => {
                setState(prev => ({ ...prev, isAuthenticated: false, currentUser: null }));
                setIsLoggingOut(false);
            }, 1000);
            return { success: true, message: 'خروج موفق' };
        } catch (e) {
            setIsLoggingOut(false);
            return { success: false, message: 'خطا' };
        }
    };

    const hasPermission = (permission: Permission): boolean => {
        if (!state.currentUser) return false;
        const userRole = state.roles.find(r => r.id === state.currentUser!.roleId);
        return userRole?.permissions.includes(permission) ?? false;
    };
    
    // --- Backup & Restore ---
    const exportData = () => {
        const fullState = { ...state, isAuthenticated: false, currentUser: null, cart: [] };
        const dataStr = JSON.stringify(fullState, null, 2);
        const blob = new Blob([dataStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `Ketabestan_Backup_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    const importData = async (file: File) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const data = JSON.parse(e.target?.result as string) as AppState;
                await api.clearAndRestoreData(data);
                await fetchData();
                showToast("✅ بازیابی موفق.");
            } catch (err) { showToast("❌ خطا در فایل."); }
        };
        reader.readAsText(file);
    };

    const cloudBackup = async () => {
        if (!navigator.onLine || !state.currentUser) return false;
        const fullState = { ...state, isAuthenticated: false, currentUser: null, cart: [] };
        const success = await api.saveCloudBackup(state.currentUser.id, fullState);
        if (success) showToast("✅ نسخه ابری با موفقیت ذخیره شد.");
        else showToast("❌ خطا در پشتیبان‌گیری ابری.");
        return success;
    };

    const cloudRestore = async () => {
        if (!navigator.onLine || !state.currentUser) return false;
        if (!window.confirm("آیا از بازیابی اطلاعات از ابر اطمینان دارید؟ تمام داده‌های فعلی جایگزین خواهند شد.")) return false;
        
        const data = await api.getCloudBackup(state.currentUser.id);
        if (data) {
            await api.clearAndRestoreData(data);
            await fetchData();
            showToast("✅ بازیابی از ابر با موفقیت انجام شد.");
            return true;
        }
        showToast("❌ هیچ نسخه پشتیبانی در ابر یافت نشد.");
        return false;
    };

    const handleSetAutoBackup = (enabled: boolean) => {
        setAutoBackupEnabled(enabled);
        localStorage.setItem('ketabestan_auto_backup', enabled.toString());
        showToast(enabled ? "✅ پشتیبان‌گیری خودکار ۲۴ ساعته فعال شد." : "⚠️ پشتیبان‌گیری خودکار غیرفعال شد.");
    };

    // --- Activities ---
    const addActivityLocal = async (type: ActivityLog['type'], description: string, user: string, refId?: string, refType?: ActivityLog['refType']) => {
        const newActivity: ActivityLog = { id: crypto.randomUUID(), type, description, timestamp: new Date().toISOString(), user, refId, refType };
        setState(prev => ({ ...prev, activities: [newActivity, ...prev.activities] }));
        try { await api.addActivity(newActivity); } catch (e) {}
        return newActivity;
    };

    // --- Users & Roles ---
    const addUser = async (userData: Omit<User, 'id'>) => {
        try {
            const newUser = await api.addUser(userData);
            setState(prev => ({ ...prev, users: [...prev.users, newUser] }));
            return { success: true, message: '✅ کاربر اضافه شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };
    const updateUser = async (userData: Partial<User> & { id: string }) => {
        try {
             await api.updateUser(userData);
             setState(prev => ({ ...prev, users: prev.users.map(u => u.id === userData.id ? { ...u, ...userData } : u) }));
             return { success: true, message: '✅ بروزرسانی شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };
    const deleteUser = async (userId: string) => {
         try {
            await api.deleteUser(userId);
            setState(prev => ({ ...prev, users: prev.users.filter(u => u.id !== userId) }));
            showToast("✅ کاربر حذف شد.");
         } catch (e) { showToast("❌ خطا."); }
    };
    const addRole = async (roleData: Omit<Role, 'id'>) => {
        try {
            const newRole = await api.addRole(roleData);
            setState(prev => ({ ...prev, roles: [...prev.roles, newRole] }));
            return { success: true, message: '✅ نقش اضافه شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };
    const updateRole = async (roleData: Role) => {
        try {
            await api.updateRole(roleData);
            setState(prev => ({ ...prev, roles: prev.roles.map(r => r.id === roleData.id ? roleData : r) }));
            return { success: true, message: '✅ نقش بروزرسانی شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };
    const deleteRole = async (roleId: string) => {
        try {
            await api.deleteRole(roleId);
            setState(prev => ({ ...prev, roles: prev.roles.filter(r => r.id !== roleId) }));
        } catch(e) { showToast("❌ خطا."); }
    };

    // --- POS ---
    const addToCart = (itemToAdd: Product | Service, type: 'product' | 'service') => {
        let success = false, message = '';
        setState(prev => {
            const existingItemIndex = prev.cart.findIndex(item => item.id === itemToAdd.id && item.type === type);
            const totalStock = type === 'product' ? (itemToAdd as Product).batches.reduce((sum, b) => sum + b.stock, 0) : Infinity;
            if (existingItemIndex > -1) {
                const updatedCart = [...prev.cart];
                const existingItem = updatedCart[existingItemIndex];
                if (type === 'product' && existingItem.quantity >= totalStock) { message = `حداکثر موجودی است.`; return prev; }
                updatedCart[existingItemIndex] = { ...existingItem, quantity: existingItem.quantity + 1 };
                success = true;
                return { ...prev, cart: updatedCart };
            } else {
                if (type === 'product' && totalStock < 1) { message = `موجودی تمام شده.`; return prev; }
                success = true;
                return { ...prev, cart: [...prev.cart, { ...itemToAdd, quantity: 1, type } as any] };
            }
        });
        return { success, message };
    };
    const updateCartItemQuantity = (itemId: string, itemType: 'product' | 'service', newQuantity: number) => {
        if (newQuantity < 0) return { success: false, message: 'نامعتبر' };
        let success = true, message = '';
        setState(prev => {
            const cart = [...prev.cart];
            const itemIndex = cart.findIndex(i => i.id === itemId && i.type === itemType);
            if (itemIndex === -1) return prev;
            if (itemType === 'product') {
                 const pInStock = prev.products.find(p => p.id === itemId);
                 const total = pInStock?.batches.reduce((sum, b) => sum + b.stock, 0) || 0;
                 if (newQuantity > total) { message = `موجودی فقط ${total} عدد است.`; cart[itemIndex] = { ...cart[itemIndex], quantity: total }; return { ...prev, cart }; }
            }
            cart[itemIndex] = { ...cart[itemIndex], quantity: newQuantity };
            if (newQuantity === 0) return { ...prev, cart: cart.filter(i => !(i.id === itemId && i.type === itemType)) };
            return { ...prev, cart };
        });
        return { success, message };
    };
    const updateCartItemFinalPrice = (itemId: string, itemType: 'product' | 'service', finalPrice: number) => {
        setState(prev => ({ ...prev, cart: prev.cart.map(item => (item.id === itemId && item.type === itemType && item.type === 'product') ? { ...item, finalPrice: finalPrice } : item) }));
    };
    const removeFromCart = (itemId: string, itemType: 'product' | 'service') => {
        setState(prev => ({ ...prev, cart: prev.cart.filter(item => !(item.id === itemId && item.type === itemType)) }));
    };
    const completeSale = async (cashier: string, customerId?: string): Promise<{ success: boolean; invoice?: SaleInvoice; message: string }> => {
        const { cart, products, editingSaleInvoiceId, customers, saleInvoices } = state;
        if (cart.length === 0) return { success: false, message: "خالی است!" };
        const subtotal = cart.reduce((total, item) => ((item.type === 'product' ? item.salePrice : item.price) * item.quantity) + total, 0);
        const totalAmount = cart.reduce((total, item) => {
            const price = (item.type === 'product' && item.finalPrice !== undefined) ? item.finalPrice : (item.type === 'product' ? item.salePrice : item.price);
            return (price * item.quantity) + total;
        }, 0);
        const stockUpdates: {batchId: string, newStock: number}[] = [];
        const saleItems: CartItem[] = [];
        const updProducts = JSON.parse(JSON.stringify(products));
        for (const item of cart) {
            if (item.type === 'service') { saleItems.push(item); continue; }
            const p = updProducts.find((p: Product) => p.id === item.id);
            let qtyToDeduct = item.quantity;
            let totalPurcValue = 0;
            p.batches.sort((a: any, b: any) => new Date(a.expiryDate || a.purchaseDate).getTime() - new Date(b.expiryDate || b.purchaseDate).getTime());
            for (const batch of p.batches) {
                if (qtyToDeduct <= 0) break;
                const deduct = Math.min(qtyToDeduct, batch.stock);
                batch.stock -= deduct; qtyToDeduct -= deduct;
                totalPurcValue += deduct * batch.purchasePrice;
                stockUpdates.push({ batchId: batch.id, newStock: batch.stock });
            }
            saleItems.push({ ...item, purchasePrice: totalPurcValue / item.quantity });
        }
        const invoiceId = editingSaleInvoiceId || generateNextId('F', saleInvoices.map(i => i.id));
        const finalInv: SaleInvoice = { id: invoiceId, type: 'sale', items: saleItems, subtotal, totalAmount, totalDiscount: subtotal - totalAmount, timestamp: new Date().toISOString(), cashier, customerId };
        let custUpdate;
        if (customerId) {
            const customer = customers.find(c => c.id === customerId)!;
            custUpdate = { id: customerId, newBalance: customer.balance + totalAmount, transaction: { id: crypto.randomUUID(), customerId, type: 'credit_sale' as const, amount: totalAmount, date: new Date().toISOString(), description: `فاکتور فروش #${invoiceId}`, invoiceId } };
        }
        try {
            if (editingSaleInvoiceId) { await api.updateSale(invoiceId, finalInv, [], [], undefined); }
            else { await api.createSale(finalInv, stockUpdates, custUpdate); }
            await fetchData();
            setState(prev => ({ ...prev, cart: [], editingSaleInvoiceId: null }));
            return { success: true, invoice: finalInv, message: 'ثبت شد.' };
        } catch (e) { return { success: false, message: 'خطا.' }; }
    };
    const beginEditSale = (invoiceId: string) => {
        const inv = state.saleInvoices.find(i => i.id === invoiceId);
        if (!inv) return { success: false, message: "یافت نشد." };
        setState(prev => ({ ...prev, editingSaleInvoiceId: invoiceId, cart: inv.items.map(i => ({ ...i } as CartItem)) }));
        return { success: true, message: "ویرایش.", customerId: inv.customerId };
    };
    const cancelEditSale = () => setState(prev => ({ ...prev, editingSaleInvoiceId: null, cart: [] }));
    const addSaleReturn = (id: string, items: any[], cashier: string) => {
        const orig = state.saleInvoices.find(i => i.id === id);
        if (!orig) return { success: false, message: "یافت نشد." };
        let total = 0;
        const detItems = items.map(ri => {
            const origI = orig.items.find(i => i.id === ri.id && i.type === ri.type);
            if (!origI) return null;
            const pr = (origI.type === 'product' && origI.finalPrice !== undefined) ? origI.finalPrice : (origI.type === 'product' ? origI.salePrice : origI.price);
            total += pr * ri.quantity;
            return { ...origI, quantity: ri.quantity };
        }).filter(Boolean) as CartItem[];
        const retInv: SaleInvoice = { id: generateNextId('R', state.saleInvoices.map(i => i.id)), type: 'return', originalInvoiceId: id, items: detItems, subtotal: total, totalAmount: total, totalDiscount: 0, timestamp: new Date().toISOString(), cashier, customerId: orig.customerId };
        api.createSaleReturn(retInv, detItems.filter(i => i.type === 'product').map(i => ({ productId: i.id, quantity: i.quantity })), orig.customerId ? { id: orig.customerId, amount: total } : undefined)
           .then(() => { fetchData(); showToast("✅ مرجوعی ثبت شد."); });
        return { success: true, message: "در حال ثبت..." };
    };
    const setInvoiceTransientCustomer = async (id: string, name: string) => {
         await api.updateSaleInvoiceMetadata(id, { original_invoice_id: name });
         setState(prev => ({ ...prev, saleInvoices: prev.saleInvoices.map(inv => inv.id === id ? { ...inv, originalInvoiceId: name } : inv) }));
    };

    // --- Inventory / Purchases ---
    const addProduct = (p: any, b: any) => { api.addProduct(p, b).then(np => { setState(prev => ({ ...prev, products: [...prev.products, np] })); showToast('✅ ذخیره شد.'); }); return { success: true, message: 'در حال ذخیره...' }; };
    const updateProduct = (p: any) => { api.updateProduct(p).then(() => { setState(prev => ({ ...prev, products: prev.products.map(x => x.id === p.id ? p : x) })); showToast('✅ ویرایش شد.'); }); return { success: true, message: 'در حال ویرایش...' }; };
    const deleteProduct = (id: string) => api.deleteProduct(id).then(() => { setState(prev => ({ ...prev, products: prev.products.filter(p => p.id !== id) })); showToast('✅ حذف شد.'); });
    const addPurchaseInvoice = (data: any) => {
        const invId = generateNextId('P', state.purchaseInvoices.map(i => i.id));
        let total = 0; const nBatches: any[] = []; const itemsNames: any[] = [];
        for (const it of data.items) {
            const pr = state.products.find(p => p.id === it.productId);
            const b = { id: crypto.randomUUID(), productId: it.productId, lotNumber: it.lotNumber, stock: it.quantity, purchasePrice: it.purchasePrice, purchaseDate: data.timestamp, expiryDate: it.expiryDate };
            nBatches.push(b); total += it.quantity * it.purchasePrice;
            itemsNames.push({ ...it, productName: pr?.name || 'ناشناس' });
        }
        if (data.currency === 'USD') total = Math.round(total * (data.exchangeRate || 1));
        const finalP: PurchaseInvoice = { id: invId, type: 'purchase', supplierId: data.supplierId, invoiceNumber: data.invoiceNumber, items: itemsNames, totalAmount: total, timestamp: data.timestamp, currency: data.currency, exchangeRate: data.exchangeRate };
        const s = state.suppliers.find(x => x.id === data.supplierId);
        const sUpd = { id: data.supplierId, newBalance: (s?.balance || 0) + total, transaction: { id: crypto.randomUUID(), supplierId: data.supplierId, type: 'purchase' as const, amount: total, date: data.timestamp, description: `فاکتور خرید #${data.invoiceNumber || invId}`, invoiceId: invId } };
        api.createPurchase(finalP, sUpd, nBatches).then(() => { fetchData(); showToast("✅ خرید ثبت شد."); });
        return { success: true, message: "در حال ثبت..." };
    };
    const beginEditPurchase = (id: string) => { setState(prev => ({ ...prev, editingPurchaseInvoiceId: id })); return { success: true, message: "ویرایش." }; };
    const cancelEditPurchase = () => setState(prev => ({ ...prev, editingPurchaseInvoiceId: null }));
    const updatePurchaseInvoice = (data: any) => {
        const orig = state.purchaseInvoices.find(i => i.id === state.editingPurchaseInvoiceId);
        if (!orig) return { success: false, message: "یافت نشد." };
        let total = 0; const itemsNames: any[] = [];
        for (const it of data.items) {
            const pr = state.products.find(p => p.id === it.productId);
            total += Number(it.quantity) * Number(it.purchasePrice);
            itemsNames.push({ ...it, productName: pr?.name || 'ناشناس' });
        }
        if (data.currency === 'USD') total = Math.round(total * (data.exchangeRate || 1));
        const upd: PurchaseInvoice = { id: state.editingPurchaseInvoiceId!, type: 'purchase', supplierId: data.supplierId, invoiceNumber: data.invoiceNumber, items: itemsNames, totalAmount: total, timestamp: data.timestamp, currency: data.currency, exchangeRate: data.exchangeRate };
        const sUpd = orig.supplierId === data.supplierId ? { id: data.supplierId, oldAmount: orig.totalAmount, newAmount: total } : undefined;
        api.updatePurchase(state.editingPurchaseInvoiceId!, upd, sUpd).then(() => { fetchData(); showToast("✅ بروزرسانی شد."); });
        return { success: true, message: "در حال بروزرسانی..." };
    };
    const addPurchaseReturn = (id: string, items: any[]) => {
        const orig = state.purchaseInvoices.find(i => i.id === id);
        if (!orig) return { success: false, message: "یافت نشد." };
        let total = 0;
        const detItems = items.map(ri => {
            const origI = orig.items.find(i => i.productId === ri.productId && i.lotNumber === ri.lotNumber);
            if (!origI) return null;
            total += origI.purchasePrice * ri.quantity;
            return { ...origI, quantity: ri.quantity };
        }).filter(Boolean) as PurchaseInvoiceItem[];
        if (orig.currency === 'USD') total = Math.round(total * (orig.exchangeRate || 1));
        const ret: PurchaseInvoice = { id: generateNextId('PR', state.purchaseInvoices.map(i => i.id)), type: 'return', originalInvoiceId: id, supplierId: orig.supplierId, invoiceNumber: `R-${orig.invoiceNumber || id}`, items: detItems, totalAmount: total, timestamp: new Date().toISOString() };
        api.createPurchaseReturn(ret, detItems.map(i => ({ productId: i.productId, quantity: i.quantity, lotNumber: i.lotNumber })), { id: orig.supplierId, amount: total })
           .then(() => { fetchData(); showToast("✅ مرجوعی خرید ثبت شد."); });
        return { success: true, message: "در حال ثبت..." };
    };

    // --- Other Entities ---
    const updateSettings = (n: any) => api.updateSettings(n).then(() => { setState(prev => ({ ...prev, storeSettings: n })); showToast("✅ تنظیمات ذخیره شد."); });
    const addService = (s: any) => api.addService(s).then(() => fetchData());
    const deleteService = (id: string) => api.deleteService(id).then(() => fetchData());
    const addSupplier = (s: any, bal: any) => api.addSupplier(s).then(ns => { if (bal) api.processPayment('supplier', ns.id, bal.amount, { id: crypto.randomUUID(), supplierId: ns.id, type: 'purchase', amount: bal.amount, date: new Date().toISOString(), description: 'تراز اول' }).then(() => fetchData()); else fetchData(); });
    const deleteSupplier = (id: string) => api.deleteSupplier(id).then(() => fetchData());
    const addCustomer = (c: any, bal: any) => api.addCustomer(c).then(nc => { if (bal) api.processPayment('customer', nc.id, bal.amount, { id: crypto.randomUUID(), customerId: nc.id, type: 'credit_sale', amount: bal.amount, date: new Date().toISOString(), description: 'تراز اول' }).then(() => fetchData()); else fetchData(); });
    const deleteCustomer = (id: string) => api.deleteCustomer(id).then(() => fetchData());
    const addEmployee = (e: any) => api.addEmployee(e).then(() => fetchData());
    const addExpense = (e: any) => api.addExpense(e).then(() => fetchData());
    const addSupplierPayment = (sid: string, amt: number, desc: string, cur: any, rate: any) => {
        const s = state.suppliers.find(x => x.id === sid)!;
        const tx = { id: crypto.randomUUID(), supplierId: sid, type: 'payment' as const, amount: amt, date: new Date().toISOString(), description: desc, currency: cur };
        api.processPayment('supplier', sid, s.balance - (cur === 'USD' ? amt * rate : amt), tx).then(() => fetchData());
        return tx;
    };
    const addCustomerPayment = (cid: string, amt: number, desc: string) => {
        const c = state.customers.find(x => x.id === cid)!;
        const tx = { id: crypto.randomUUID(), customerId: cid, type: 'payment' as const, amount: amt, date: new Date().toISOString(), description: desc };
        api.processPayment('customer', cid, c.balance - amt, tx).then(() => fetchData());
        return tx;
    };
    const addEmployeeAdvance = (eid: string, amt: number) => {
        const e = state.employees.find(x => x.id === eid)!;
        api.processPayment('employee', eid, e.balance + amt, { id: crypto.randomUUID(), employeeId: eid, type: 'advance' as const, amount: amt, date: new Date().toISOString(), description: 'مساعده' }).then(() => fetchData());
    };
    const processAndPaySalaries = () => {
        let tot = 0; const txs: any[] = [];
        state.employees.forEach(e => {
            const net = e.monthlySalary - e.balance;
            if (net > 0) { txs.push({ id: crypto.randomUUID(), employeeId: e.id, type: 'salary_payment', amount: net, date: new Date().toISOString(), description: 'حقوق' }); tot += net; }
        });
        if (tot === 0) return { success: false, message: 'موردی نیست.' };
        api.processPayroll(state.employees.map(e => ({ id: e.id, balance: 0 as const })), txs, { id: crypto.randomUUID(), category: 'salary', description: 'حقوق', amount: tot, date: new Date().toISOString() }).then(() => fetchData());
        return { success: true, message: 'در حال پردازش...' };
    };

    if (isLoading) return <div className="flex items-center justify-center h-screen text-xl font-bold text-blue-600">در حال دریافت اطلاعات...</div>;

    return <AppContext.Provider value={{
        ...state, showToast, isLoading, isLoggingOut, login, signup, logout, hasPermission, addUser, updateUser, deleteUser, addRole, updateRole, deleteRole, exportData, importData,
        cloudBackup, cloudRestore, autoBackupEnabled, setAutoBackupEnabled: handleSetAutoBackup,
        addProduct, updateProduct, deleteProduct, addToCart, updateCartItemQuantity, updateCartItemFinalPrice, removeFromCart, completeSale,
        beginEditSale, cancelEditSale, addSaleReturn, addPurchaseInvoice, beginEditPurchase, cancelEditPurchase, updatePurchaseInvoice, addPurchaseReturn,
        updateSettings, addService, deleteService, addSupplier, deleteSupplier, addSupplierPayment, addCustomer, deleteCustomer, addCustomerPayment,
        addEmployee, addEmployeeAdvance, processAndPaySalaries, addExpense, setInvoiceTransientCustomer
    }}>{children}</AppContext.Provider>;
};

export const useAppContext = (): AppContextType => {
    const context = useContext(AppContext);
    if (context === undefined) throw new Error('useAppContext must be used within an AppProvider');
    return context;
};
